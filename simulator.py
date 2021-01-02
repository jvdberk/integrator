from multiprocessing import Process, Pipe, shared_memory  # v3.8
import numpy as np
import timeit
from numba import njit
import h5py


def simulator(mem_name, layout, clen, conn, integrator, *args):
    # attach an existing shared memory block
    shm_there = shared_memory.SharedMemory(mem_name)
    out = np.ndarray(layout[:-1], dtype=layout[-1], buffer=shm_there.buf)

    run = integrator(*args)

    for rdx, msg in enumerate(run):
        # place in shared memory
        # wait if last mesg < rdx -blen
        out[rdx % (layout[0])][:] = msg

        # every block shed stored values
        #print(rdx, clen, (rdx+1)%clen, rdx%(layout[0]))
        if (rdx+1) % clen == 0:
            if conn.recv():
                # output process is ready
                conn.send((rdx) % layout[0]+1)

    # None as sentinel
    conn.send(None)
    # close pipe
    conn.close()

    # close shared memory end
    shm_there.close()


def main():
    from lorenz import lorenz as plorenz

    # initial state variables
    y0 = (1., 1., 1.)
    # integration parameters
    t0 = 0      # time at t = t0
    te = 20     # end time
    dt = 1.e-7  # integration time step
    bt = 0.01   # delta time output

    # shared memory space to place (buffered) output
    blen = 10  # total buffer length
    clen = 5  # output block length

    stshp = (4,)  # shape of state space vector: time + R^3
    shape = (blen,) + stshp

    size = blen
    for s in shape[1:]:
        size *= s

    bwidth = np.array([0], dtype=np.float64).nbytes

    nbytes = bwidth * size

    # claim shared memory space
    shm_here = shared_memory.SharedMemory(create=True, size=nbytes)
    # NumPy array of doubles on that space
    out = np.ndarray(shape, dtype=np.float64, buffer=shm_here.buf)

    layout = shape + (out.dtype,)
    print('Buffer memory layout', ' * '.join(map(str, layout)))

    # pipes are 1 - 1 connections (no multiplexing) full duplex
    here, there = Pipe()

    # jit the pure python function
    lorenz = njit(plorenz)

    # define process of simulator parametrised with args
    # shared name, mem layout, chunk, pipe
    p = Process(target=simulator, args=(shm_here.name, layout, clen, there
                                   # integrator
                                                     , euler
                                   # args to integrator
                                   # function to integrate
                                   # initial values on func.
                                                     , t0, te, dt, bt
                                                     , lorenz, *y0
                                        )
                )

    # start process p and run its target
    p.start()

    # create group
    out_fn = 'lorenz.h5'
    with h5py.File(out_fn, mode='w') as root:
        # define group
        grp = root.create_group("lorenz")

        # define time component
        step = grp.create_dataset('step', dtype=out.dtype)
        step.make_scale('step units')

        # state
        state = grp.create_dataset('state', dtype=out.dtype)
        state.make_scale('state units')

        # and dataset within the group in terms of these scales
        root["run"] = grp.create_dataset(
            "values", shape=shape, maxshape=(None,) + stshp, dtype=out.dtype)
        run = root['run']

        # attach scales
        run.dims[0].label = 'step'
        run.dims[0].attach_scale(step)

        run.dims[1].label = 'state'
        run.dims[1].attach_scale(state)

        print('Datasets:', list(root))
        print('dimension labels:', [dim.label for dim in root['run'].dims])

        # record initial state
        run[0] = (t0, *y0)

        here.send(1)
        # wait for signals of last index
        msg = here.recv()
        rdx = 1
        while msg is not None:
            # None is sentinel for output done

            # resize run if needed - increment with block length
            if (rdx + clen) > len(run):
                run.resize((len(run) + blen,) + stshp)

            run[rdx:rdx+clen] = out[(msg-clen):msg]
            rdx += clen

            # done with output processing
            here.send(True)

            # await new output block
            msg = here.recv()

        # shrink to fit
        run.resize((rdx,) + stshp)

    # join the process back into the main process (caller)
    p.join()

    shm_here.close()  # Close each SharedMemory instance
    shm_here.unlink()  # unlink once per instance (here, not there)


def gen_euler(nstate=1):
    # construct function as closure over a dynamics function (f)
    # and integration parameters
    # code generation from formatted strings

    # object (variables)
    fargs = tuple(('y0'+str(n) for n in range(nstate)))
    fvars = tuple(('y'+str(n) for n in range(nstate)))
    dvars = tuple(('dy'+str(n) for n in range(nstate)))

    # map objects to read-out function
    rdt = "yield (t, {fvars})".format(fvars=', '.join(fvars))

    # dynamics
    dyn = "f(t, {fvars})".format(fvars=', '.join(fvars))

    # Euler update
    upd = "{evars}".format(evars=', '.join(map(lambda f, s: f+'+'+s+'*h'
                                             , fvars, dvars
                                               )
                                           ), fvars=', '.join(fvars)
                           )

    # loop and read in of objects
    begin = """
@njit
def euler(t0, te, h, H, f, {fargs}):
    t, {fvars} = t0, {fargs}

    # internal object
    t_1 = 0

    while t <= te:
        # output state if time limit not reached at H intervals
        # includes initial state
        if (t - t_1 - h)/H >= 1:
            {rdt}
            t_1 = t
    """.format(fargs=', '.join(fargs)
             , fvars=', '.join(fvars)
             , dvars=', '.join(dvars)
             , rdt=rdt
               )

    # fvars and dvars objects (variables)
    # dvars are mapped from parametrised vector field dyn (dynamics)
    # going from Time x State -> Tangent-space(State)
    # fvars are mapped from the discretisation taking the continuous dynamics
    # to the discrete update function (monoidal, compositional - important!)
    euler = """
        t += h
        # Euler updates (internal objects)
        {dvars} = {dyn}
        
        # map from continuous to discrete dynamical system
        # and apply to state space
        {fvars} = {upd}
    """.format(fvars=', '.join(fvars)
             , dvars=', '.join(dvars)
             , dyn=dyn
             , upd=upd
               )

    end = """
    else:
        # output final state
        {rdt}
        """.format(rdt=rdt)

    return begin+euler+end


# define function from string definition
# "euler" placed in namespace (function object)
exec(gen_euler(3))

if __name__ == '__main__':
    t01 = timeit.timeit(main, number=1)

    print(t01)

    print(gen_euler(3))
