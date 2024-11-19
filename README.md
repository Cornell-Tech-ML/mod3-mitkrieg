# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Output from project/parallel_check.py

```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py (163)
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        in_storage: Storage,                                             |
        in_shape: Shape,                                                 |
        in_strides: Strides,                                             |
    ) -> None:                                                           |
        # TODO: Implement for Task 3.1.                                  |
        # if stride-aligned, avoid indexing                              |
        out_size = np.prod(out_shape)------------------------------------| #2
        if np.array_equal(in_shape, out_shape) and np.array_equal(       |
            in_strides, out_strides                                      |
        ):                                                               |
            for i in prange(out_size):-----------------------------------| #3
                out[i] = fn(in_storage[i])                               |
        else:                                                            |
            for i in prange(out_size):-----------------------------------| #4
                out_idx = np.zeros(len(out_shape), dtype=np.int32)-------| #0
                in_idx = np.zeros(len(in_shape), dtype=np.int32)---------| #1
                to_index(i, out_shape, out_idx)                          |
                broadcast_index(out_idx, out_shape, in_shape, in_idx)    |
                pos = index_to_position(in_idx, in_strides)              |
                out[i] = fn(in_storage[pos])                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #2, #3, #4, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--4 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (serial)
   +--1 (serial)



Parallel region 0 (loop #4) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#4).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(181) is hoisted out of the parallel loop labelled #4 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(182) is hoisted out of the parallel loop labelled #4 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_idx = np.zeros(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(214)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py (214)
-----------------------------------------------------------------------|loop #ID
    def _zip(                                                          |
        out: Storage,                                                  |
        out_shape: Shape,                                              |
        out_strides: Strides,                                          |
        a_storage: Storage,                                            |
        a_shape: Shape,                                                |
        a_strides: Strides,                                            |
        b_storage: Storage,                                            |
        b_shape: Shape,                                                |
        b_strides: Strides,                                            |
    ) -> None:                                                         |
        # TODO: Implement for Task 3.1.                                |
                                                                       |
        # if stride-aligned, avoid indexing                            |
        if (                                                           |
            np.array_equal(a_strides, out_strides)                     |
            and np.array_equal(b_strides, out_strides)                 |
            and np.array_equal(a_shape, out_shape)                     |
            and np.array_equal(b_shape, out_shape)                     |
        ):                                                             |
            for i in prange(len(out)):---------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                |
        else:                                                          |
            for i in prange(len(out)):---------------------------------| #9
                a_idx = np.zeros(len(a_shape), dtype=np.int32)---------| #5
                b_idx = np.zeros(len(b_shape), dtype=np.int32)---------| #6
                out_idx = np.zeros(len(out_shape), dtype=np.int32)-----| #7
                                                                       |
                to_index(i, out_shape, out_idx)                        |
                broadcast_index(out_idx, out_shape, a_shape, a_idx)    |
                broadcast_index(out_idx, out_shape, b_shape, b_idx)    |
                                                                       |
                a_pos = index_to_position(a_idx, a_strides)            |
                b_pos = index_to_position(b_idx, b_strides)            |
                out[i] = fn(a_storage[a_pos], b_storage[b_pos])        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #8, #9, #5, #6, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--9 is a parallel loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--5 (parallel)
   +--6 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--9 (parallel)
   +--5 (serial)
   +--6 (serial)
   +--7 (serial)



Parallel region 0 (loop #9) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#9).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(238) is hoisted out of the parallel loop labelled #9 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_idx = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(239) is hoisted out of the parallel loop labelled #9 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_idx = np.zeros(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(240) is hoisted out of the parallel loop labelled #9 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_idx = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(274)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py (274)
-----------------------------------------------------------------------|loop #ID
    def _reduce(                                                       |
        out: Storage,                                                  |
        out_shape: Shape,                                              |
        out_strides: Strides,                                          |
        a_storage: Storage,                                            |
        a_shape: Shape,                                                |
        a_strides: Strides,                                            |
        reduce_dim: int,                                               |
    ) -> None:                                                         |
        # TODO: Implement for Task 3.1.                                |
        dim_len = a_shape[reduce_dim]                                  |
        start = out[0]                                                 |
                                                                       |
        for i in prange(np.prod(out_shape)):---------------------------| #13, 12
            out_index = np.zeros(len(out_shape), dtype=np.int32)-------| #10
            a_index = np.zeros(len(a_shape), dtype=np.int32)-----------| #11
            to_index(i, out_shape, out_index)                          |
                                                                       |
            accumulator = start                                        |
                                                                       |
            for j in range(dim_len):                                   |
                for k in range(len(out_shape)):                        |
                    a_index[k] = out_index[k]                          |
                a_index[reduce_dim] = j                                |
                                                                       |
                a_pos = index_to_position(a_index, a_strides)          |
                accumulator = fn(accumulator, a_storage[a_pos])        |
                                                                       |
            out[i] = accumulator                                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #12, #13, #10, #11).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--13 is a parallel loop
   +--10 --> rewritten as a serial loop
   +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--10 (parallel)
   +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--10 (serial)
   +--11 (serial)



Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(288) is hoisted out of the parallel loop labelled #13 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(289) is hoisted out of the parallel loop labelled #13 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py
(307)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/mitchellkrieger/Documents/GitHub/mod3-mitkrieg/minitorch/fast_ops.py (307)
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              |
    out: Storage,                                                                         |
    out_shape: Shape,                                                                     |
    out_strides: Strides,                                                                 |
    a_storage: Storage,                                                                   |
    a_shape: Shape,                                                                       |
    a_strides: Strides,                                                                   |
    b_storage: Storage,                                                                   |
    b_shape: Shape,                                                                       |
    b_strides: Strides,                                                                   |
) -> None:                                                                                |
    """NUMBA tensor matrix multiply function.                                             |
                                                                                          |
    Should work for any tensor shapes that broadcast as long as                           |
                                                                                          |
    ```                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                     |
    ```                                                                                   |
                                                                                          |
    Optimizations:                                                                        |
                                                                                          |
    * Outer loop in parallel                                                              |
    * No index buffers or function calls                                                  |
    * Inner loop should have no global writes, 1 multiply.                                |
                                                                                          |
                                                                                          |
    Args:                                                                                 |
    ----                                                                                  |
        out (Storage): storage for `out` tensor                                           |
        out_shape (Shape): shape for `out` tensor                                         |
        out_strides (Strides): strides for `out` tensor                                   |
        a_storage (Storage): storage for `a` tensor                                       |
        a_shape (Shape): shape for `a` tensor                                             |
        a_strides (Strides): strides for `a` tensor                                       |
        b_storage (Storage): storage for `b` tensor                                       |
        b_shape (Shape): shape for `b` tensor                                             |
        b_strides (Strides): strides for `b` tensor                                       |
                                                                                          |
    Returns:                                                                              |
    -------                                                                               |
        None : Fills in `out`                                                             |
                                                                                          |
    """                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                |
                                                                                          |
    # TODO: Implement for Task 3.2.                                                       |
    assert a_shape[-1] == b_shape[-2]                                                     |
                                                                                          |
    slice_2d = out_shape[-1] * out_shape[-2]                                              |
                                                                                          |
    for i in prange(len(out)):------------------------------------------------------------| #14
        # get each position                                                               |
        batch = i // slice_2d                                                             |
        row = (i % slice_2d) // out_shape[-1]                                             |
        col = i % out_shape[-1]                                                           |
                                                                                          |
        # find where a and b begin                                                        |
        a = batch * a_batch_stride + row * a_strides[1]                                   |
        b = batch * b_batch_stride + col * b_strides[2]                                   |
                                                                                          |
        # multipy and accumulate to perform matmul                                        |
        tmp = 0                                                                           |
        for position in range(a_shape[-1]):                                               |
            tmp += (                                                                      |
                a_storage[a + position * a_strides[2]]                                    |
                * b_storage[b + position * b_strides[1]]                                  |
            )                                                                             |
        out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]    |
        out[out_pos] = tmp                                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #14).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

## Model training

## Model Training Logs

### Simple

Dataset: Simple
Backend: CPU
Hidden: 100
LR: 0.05

```
Epoch:     0 || Loss:  8.32548 || Correct    25 || Time per epoch:  18.39554
Epoch:    10 || Loss:  6.47043 || Correct    40 || Time per epoch:  0.09881
Epoch:    20 || Loss:  3.99787 || Correct    39 || Time per epoch:  0.09808
Epoch:    30 || Loss:  2.95462 || Correct    39 || Time per epoch:  0.09782
Epoch:    40 || Loss:  5.75990 || Correct    46 || Time per epoch:  0.09822
Epoch:    50 || Loss:  2.24706 || Correct    48 || Time per epoch:  0.14127
Epoch:    60 || Loss:  1.54707 || Correct    44 || Time per epoch:  0.15890
Epoch:    70 || Loss:  2.59027 || Correct    49 || Time per epoch:  0.09943
Epoch:    80 || Loss:  3.07012 || Correct    49 || Time per epoch:  0.10017
Epoch:    90 || Loss:  3.73035 || Correct    45 || Time per epoch:  0.09817
Epoch:   100 || Loss:  2.40822 || Correct    50 || Time per epoch:  0.09741
Epoch:   110 || Loss:  2.12735 || Correct    49 || Time per epoch:  0.09832
Epoch:   120 || Loss:  1.31102 || Correct    50 || Time per epoch:  0.09887
Epoch:   130 || Loss:  0.48581 || Correct    48 || Time per epoch:  0.09799
Epoch:   140 || Loss:  2.67849 || Correct    46 || Time per epoch:  0.09762
Epoch:   150 || Loss:  0.86275 || Correct    48 || Time per epoch:  0.09803
Epoch:   160 || Loss:  1.35992 || Correct    50 || Time per epoch:  0.09761
Epoch:   170 || Loss:  0.99190 || Correct    50 || Time per epoch:  0.17720
Epoch:   180 || Loss:  0.74015 || Correct    48 || Time per epoch:  0.12902
Epoch:   190 || Loss:  3.16690 || Correct    46 || Time per epoch:  0.09795
Epoch:   200 || Loss:  1.68249 || Correct    50 || Time per epoch:  0.09778
Epoch:   210 || Loss:  0.82470 || Correct    50 || Time per epoch:  0.09736
Epoch:   220 || Loss:  0.88740 || Correct    50 || Time per epoch:  0.10204
Epoch:   230 || Loss:  0.74309 || Correct    49 || Time per epoch:  0.09836
Epoch:   240 || Loss:  0.90257 || Correct    50 || Time per epoch:  0.09738
Epoch:   250 || Loss:  0.85901 || Correct    50 || Time per epoch:  0.09770
Epoch:   260 || Loss:  0.62612 || Correct    50 || Time per epoch:  0.09770
Epoch:   270 || Loss:  0.45352 || Correct    49 || Time per epoch:  0.09899
Epoch:   280 || Loss:  0.62163 || Correct    50 || Time per epoch:  0.13472
Epoch:   290 || Loss:  0.10163 || Correct    49 || Time per epoch:  0.16312
Epoch:   300 || Loss:  0.36479 || Correct    50 || Time per epoch:  0.09711
Epoch:   310 || Loss:  0.48168 || Correct    50 || Time per epoch:  0.09712
Epoch:   320 || Loss:  0.71765 || Correct    50 || Time per epoch:  0.09764
Epoch:   330 || Loss:  0.74919 || Correct    50 || Time per epoch:  0.09802
Epoch:   340 || Loss:  0.96047 || Correct    50 || Time per epoch:  0.09848
Epoch:   350 || Loss:  0.52161 || Correct    50 || Time per epoch:  0.09754
Epoch:   360 || Loss:  0.31385 || Correct    50 || Time per epoch:  0.09812
Epoch:   370 || Loss:  0.31923 || Correct    50 || Time per epoch:  0.09864
Epoch:   380 || Loss:  0.30227 || Correct    50 || Time per epoch:  0.09719
Epoch:   390 || Loss:  1.00093 || Correct    50 || Time per epoch:  0.09703
Epoch:   400 || Loss:  0.76221 || Correct    50 || Time per epoch:  0.16934
Epoch:   410 || Loss:  0.08206 || Correct    50 || Time per epoch:  0.13446
Epoch:   420 || Loss:  0.52891 || Correct    50 || Time per epoch:  0.09726
Epoch:   430 || Loss:  0.38505 || Correct    50 || Time per epoch:  0.09814
Epoch:   440 || Loss:  0.23398 || Correct    50 || Time per epoch:  0.09814
Epoch:   450 || Loss:  0.23576 || Correct    50 || Time per epoch:  0.09832
Epoch:   460 || Loss:  0.29704 || Correct    50 || Time per epoch:  0.09830
Epoch:   470 || Loss:  0.30774 || Correct    50 || Time per epoch:  0.09857
Epoch:   480 || Loss:  0.02018 || Correct    50 || Time per epoch:  0.09722
Epoch:   490 || Loss:  0.00950 || Correct    50 || Time per epoch:  0.09716
```


Dataset: Simple
Backend: GPU
Hidden: 100
LR: 0.05

```
Epoch:     0 || Loss:  6.75284 || Correct    31 || Time per epoch:  4.77605
Epoch:    10 || Loss:  5.58277 || Correct    41 || Time per epoch:  1.97466
Epoch:    20 || Loss:  6.68693 || Correct    38 || Time per epoch:  1.94805
Epoch:    30 || Loss:  2.52563 || Correct    47 || Time per epoch:  1.87270
Epoch:    40 || Loss:  3.09321 || Correct    49 || Time per epoch:  1.94237
Epoch:    50 || Loss:  2.38445 || Correct    48 || Time per epoch:  1.86161
Epoch:    60 || Loss:  2.05923 || Correct    50 || Time per epoch:  1.94432
Epoch:    70 || Loss:  1.20019 || Correct    47 || Time per epoch:  1.86829
Epoch:    80 || Loss:  2.14480 || Correct    50 || Time per epoch:  1.95073
Epoch:    90 || Loss:  2.35514 || Correct    50 || Time per epoch:  1.92010
Epoch:   100 || Loss:  0.76116 || Correct    48 || Time per epoch:  1.89306
Epoch:   110 || Loss:  0.76636 || Correct    49 || Time per epoch:  1.94913
Epoch:   120 || Loss:  1.21298 || Correct    50 || Time per epoch:  1.86073
Epoch:   130 || Loss:  0.87785 || Correct    50 || Time per epoch:  1.94288
Epoch:   140 || Loss:  0.89645 || Correct    50 || Time per epoch:  1.85993
Epoch:   150 || Loss:  1.14349 || Correct    50 || Time per epoch:  1.94711
Epoch:   160 || Loss:  0.88506 || Correct    50 || Time per epoch:  1.86889
Epoch:   170 || Loss:  1.31130 || Correct    50 || Time per epoch:  1.92485
Epoch:   180 || Loss:  0.46336 || Correct    50 || Time per epoch:  1.94373
Epoch:   190 || Loss:  0.37956 || Correct    50 || Time per epoch:  1.86076
Epoch:   200 || Loss:  0.51409 || Correct    50 || Time per epoch:  1.94921
Epoch:   210 || Loss:  0.80199 || Correct    50 || Time per epoch:  1.86188
Epoch:   220 || Loss:  0.21237 || Correct    50 || Time per epoch:  1.93738
Epoch:   230 || Loss:  0.90014 || Correct    50 || Time per epoch:  1.86324
Epoch:   240 || Loss:  0.23754 || Correct    50 || Time per epoch:  1.95168
Epoch:   250 || Loss:  0.48261 || Correct    50 || Time per epoch:  1.92656
Epoch:   260 || Loss:  0.58314 || Correct    50 || Time per epoch:  1.87206
Epoch:   270 || Loss:  0.24448 || Correct    50 || Time per epoch:  1.96329
Epoch:   280 || Loss:  0.10696 || Correct    50 || Time per epoch:  1.87595
Epoch:   290 || Loss:  0.03216 || Correct    50 || Time per epoch:  1.94253
Epoch:   300 || Loss:  0.44955 || Correct    50 || Time per epoch:  1.86138
Epoch:   310 || Loss:  0.09414 || Correct    50 || Time per epoch:  1.95082
Epoch:   320 || Loss:  0.46210 || Correct    50 || Time per epoch:  1.92245
Epoch:   330 || Loss:  0.46836 || Correct    50 || Time per epoch:  1.87132
Epoch:   340 || Loss:  0.07113 || Correct    50 || Time per epoch:  1.95224
Epoch:   350 || Loss:  0.61147 || Correct    50 || Time per epoch:  1.86682
Epoch:   360 || Loss:  0.49413 || Correct    50 || Time per epoch:  1.94391
Epoch:   370 || Loss:  0.24420 || Correct    50 || Time per epoch:  1.86453
Epoch:   380 || Loss:  0.40807 || Correct    50 || Time per epoch:  1.92774
Epoch:   390 || Loss:  0.22841 || Correct    50 || Time per epoch:  1.86420
Epoch:   400 || Loss:  0.20728 || Correct    50 || Time per epoch:  1.92873
Epoch:   410 || Loss:  0.05500 || Correct    50 || Time per epoch:  1.94213
Epoch:   420 || Loss:  0.61322 || Correct    50 || Time per epoch:  1.85387
Epoch:   430 || Loss:  0.24672 || Correct    50 || Time per epoch:  1.95433
Epoch:   440 || Loss:  0.13920 || Correct    50 || Time per epoch:  1.85722
Epoch:   450 || Loss:  0.35801 || Correct    50 || Time per epoch:  1.93630
Epoch:   460 || Loss:  0.03450 || Correct    50 || Time per epoch:  1.86143
Epoch:   470 || Loss:  0.11887 || Correct    50 || Time per epoch:  1.94299
Epoch:   480 || Loss:  0.18061 || Correct    50 || Time per epoch:  1.89205
Epoch:   490 || Loss:  0.10301 || Correct    50 || Time per epoch:  1.90425
```

Dataset: Simple
Backend: CPU
Hidden: 500
LR: 0.05

```
Epoch:     0 || Loss:  26.63657 || Correct    39 || Time per epoch:  18.94309
Epoch:    10 || Loss:  4.42665 || Correct    44 || Time per epoch:  0.83532
Epoch:    20 || Loss:  1.08549 || Correct    49 || Time per epoch:  0.82688
Epoch:    30 || Loss:  2.25921 || Correct    48 || Time per epoch:  0.74359
Epoch:    40 || Loss:  2.47335 || Correct    44 || Time per epoch:  0.84621
Epoch:    50 || Loss:  0.73759 || Correct    49 || Time per epoch:  0.79610
Epoch:    60 || Loss:  0.32806 || Correct    49 || Time per epoch:  0.77416
Epoch:    70 || Loss:  1.79367 || Correct    49 || Time per epoch:  0.84081
Epoch:    80 || Loss:  0.80014 || Correct    49 || Time per epoch:  0.78349
Epoch:    90 || Loss:  0.09830 || Correct    47 || Time per epoch:  0.79658
Epoch:   100 || Loss:  1.13797 || Correct    49 || Time per epoch:  0.84541
Epoch:   110 || Loss:  0.03126 || Correct    49 || Time per epoch:  0.76781
Epoch:   120 || Loss:  0.13515 || Correct    49 || Time per epoch:  0.80149
Epoch:   130 || Loss:  0.98313 || Correct    50 || Time per epoch:  0.84891
Epoch:   140 || Loss:  1.70810 || Correct    47 || Time per epoch:  0.75008
Epoch:   150 || Loss:  0.00314 || Correct    48 || Time per epoch:  0.81528
Epoch:   160 || Loss:  1.22005 || Correct    50 || Time per epoch:  0.84248
Epoch:   170 || Loss:  1.28310 || Correct    49 || Time per epoch:  0.72654
Epoch:   180 || Loss:  0.05553 || Correct    49 || Time per epoch:  0.83338
Epoch:   190 || Loss:  0.22367 || Correct    49 || Time per epoch:  0.83317
Epoch:   200 || Loss:  0.76648 || Correct    49 || Time per epoch:  0.71882
Epoch:   210 || Loss:  0.28571 || Correct    49 || Time per epoch:  0.82922
Epoch:   220 || Loss:  0.96858 || Correct    49 || Time per epoch:  0.82905
Epoch:   230 || Loss:  0.47136 || Correct    49 || Time per epoch:  0.72687
Epoch:   240 || Loss:  1.50203 || Correct    48 || Time per epoch:  0.84422
Epoch:   250 || Loss:  0.76940 || Correct    49 || Time per epoch:  0.83380
Epoch:   260 || Loss:  0.01824 || Correct    49 || Time per epoch:  0.72818
Epoch:   270 || Loss:  0.00545 || Correct    50 || Time per epoch:  0.83683
Epoch:   280 || Loss:  0.46591 || Correct    50 || Time per epoch:  0.83258
Epoch:   290 || Loss:  0.06448 || Correct    49 || Time per epoch:  0.72664
Epoch:   300 || Loss:  0.07227 || Correct    48 || Time per epoch:  0.83250
Epoch:   310 || Loss:  1.08556 || Correct    49 || Time per epoch:  0.83873
Epoch:   320 || Loss:  0.55761 || Correct    50 || Time per epoch:  0.72817
Epoch:   330 || Loss:  0.77213 || Correct    49 || Time per epoch:  0.83462
Epoch:   340 || Loss:  0.12404 || Correct    49 || Time per epoch:  0.83629
Epoch:   350 || Loss:  0.47125 || Correct    49 || Time per epoch:  0.73288
Epoch:   360 || Loss:  0.11341 || Correct    49 || Time per epoch:  0.83604
Epoch:   370 || Loss:  0.53074 || Correct    49 || Time per epoch:  0.80545
Epoch:   380 || Loss:  0.02952 || Correct    49 || Time per epoch:  0.74958
Epoch:   390 || Loss:  0.08037 || Correct    50 || Time per epoch:  0.83091
Epoch:   400 || Loss:  0.10601 || Correct    49 || Time per epoch:  0.78214
Epoch:   410 || Loss:  0.62870 || Correct    49 || Time per epoch:  0.75907
Epoch:   420 || Loss:  0.22065 || Correct    50 || Time per epoch:  0.82327
Epoch:   430 || Loss:  0.74018 || Correct    49 || Time per epoch:  0.76078
Epoch:   440 || Loss:  0.65406 || Correct    49 || Time per epoch:  0.77645
Epoch:   450 || Loss:  0.23989 || Correct    50 || Time per epoch:  0.82311
Epoch:   460 || Loss:  0.06309 || Correct    50 || Time per epoch:  0.71788
Epoch:   470 || Loss:  0.00118 || Correct    50 || Time per epoch:  0.82443
Epoch:   480 || Loss:  0.23151 || Correct    50 || Time per epoch:  0.82296
Epoch:   490 || Loss:  0.52680 || Correct    50 || Time per epoch:  0.72478
```

Dataset: Simple
Backend: GPU
Hidden: 500
LR: 0.05

```
Epoch:     0 || Loss:  42.14486 || Correct    35 || Time per epoch:  5.52751
Epoch:    10 || Loss:  1.49197 || Correct    48 || Time per epoch:  2.54132
Epoch:    20 || Loss:  0.27354 || Correct    50 || Time per epoch:  2.51078
Epoch:    30 || Loss:  0.11967 || Correct    50 || Time per epoch:  2.51609
Epoch:    40 || Loss:  0.20061 || Correct    50 || Time per epoch:  2.48845
Epoch:    50 || Loss:  0.04813 || Correct    50 || Time per epoch:  2.51117
Epoch:    60 || Loss:  0.43111 || Correct    50 || Time per epoch:  2.50816
Epoch:    70 || Loss:  0.03283 || Correct    50 || Time per epoch:  2.52371
Epoch:    80 || Loss:  0.18903 || Correct    50 || Time per epoch:  2.51416
Epoch:    90 || Loss:  0.06027 || Correct    50 || Time per epoch:  2.50389
Epoch:   100 || Loss:  0.24614 || Correct    50 || Time per epoch:  2.51252
Epoch:   110 || Loss:  0.02680 || Correct    50 || Time per epoch:  2.50932
Epoch:   120 || Loss:  0.04301 || Correct    50 || Time per epoch:  2.52006
Epoch:   130 || Loss:  0.06987 || Correct    50 || Time per epoch:  2.54098
Epoch:   140 || Loss:  0.09600 || Correct    50 || Time per epoch:  2.54096
Epoch:   150 || Loss:  0.06840 || Correct    50 || Time per epoch:  2.51686
Epoch:   160 || Loss:  0.09925 || Correct    50 || Time per epoch:  2.51465
Epoch:   170 || Loss:  0.03967 || Correct    50 || Time per epoch:  2.51075
Epoch:   180 || Loss:  0.07584 || Correct    50 || Time per epoch:  2.50160
Epoch:   190 || Loss:  0.07697 || Correct    50 || Time per epoch:  2.53213
Epoch:   200 || Loss:  0.04652 || Correct    50 || Time per epoch:  2.51394
Epoch:   210 || Loss:  0.11926 || Correct    50 || Time per epoch:  2.48633
Epoch:   220 || Loss:  0.13272 || Correct    50 || Time per epoch:  2.49763
Epoch:   230 || Loss:  0.05600 || Correct    50 || Time per epoch:  2.53263
Epoch:   240 || Loss:  0.04916 || Correct    50 || Time per epoch:  2.49847
Epoch:   250 || Loss:  0.02100 || Correct    50 || Time per epoch:  2.48631
Epoch:   260 || Loss:  0.01018 || Correct    50 || Time per epoch:  2.53555
Epoch:   270 || Loss:  0.07433 || Correct    50 || Time per epoch:  2.49700
Epoch:   280 || Loss:  0.08836 || Correct    50 || Time per epoch:  2.51607
Epoch:   290 || Loss:  0.05976 || Correct    50 || Time per epoch:  2.53303
Epoch:   300 || Loss:  0.08221 || Correct    50 || Time per epoch:  2.52686
Epoch:   310 || Loss:  0.03649 || Correct    50 || Time per epoch:  2.51033
Epoch:   320 || Loss:  0.02947 || Correct    50 || Time per epoch:  2.49012
Epoch:   330 || Loss:  0.03893 || Correct    50 || Time per epoch:  2.48588
Epoch:   340 || Loss:  0.04478 || Correct    50 || Time per epoch:  2.49812
Epoch:   350 || Loss:  0.04120 || Correct    50 || Time per epoch:  2.49944
Epoch:   360 || Loss:  0.02590 || Correct    50 || Time per epoch:  2.49253
Epoch:   370 || Loss:  0.02346 || Correct    50 || Time per epoch:  2.48747
Epoch:   380 || Loss:  0.06421 || Correct    50 || Time per epoch:  2.48179
Epoch:   390 || Loss:  0.01049 || Correct    50 || Time per epoch:  2.48777
Epoch:   400 || Loss:  0.06296 || Correct    50 || Time per epoch:  2.48992
Epoch:   410 || Loss:  0.02813 || Correct    50 || Time per epoch:  2.49037
Epoch:   420 || Loss:  0.07447 || Correct    50 || Time per epoch:  2.48463
Epoch:   430 || Loss:  0.06173 || Correct    50 || Time per epoch:  2.49808
Epoch:   440 || Loss:  0.04518 || Correct    50 || Time per epoch:  2.49432
Epoch:   450 || Loss:  0.07053 || Correct    50 || Time per epoch:  2.47132
Epoch:   460 || Loss:  0.03408 || Correct    50 || Time per epoch:  2.48783
Epoch:   470 || Loss:  0.02356 || Correct    50 || Time per epoch:  2.49280
Epoch:   480 || Loss:  0.06007 || Correct    50 || Time per epoch:  2.49457
Epoch:   490 || Loss:  0.02870 || Correct    50 || Time per epoch:  2.51160
```

### Split

Dataset: Split
Backend: CPU
Hidden: 100
LR: 0.05

```
Epoch:     0 || Loss:  4.07173 || Correct    32 || Time per epoch:  18.16308
Epoch:    10 || Loss:  4.52326 || Correct    39 || Time per epoch:  0.09813
Epoch:    20 || Loss:  5.17273 || Correct    46 || Time per epoch:  0.09768
Epoch:    30 || Loss:  3.45568 || Correct    45 || Time per epoch:  0.09738
Epoch:    40 || Loss:  1.34912 || Correct    47 || Time per epoch:  0.09745
Epoch:    50 || Loss:  2.84589 || Correct    50 || Time per epoch:  0.17305
Epoch:    60 || Loss:  1.40761 || Correct    47 || Time per epoch:  0.13012
Epoch:    70 || Loss:  1.40857 || Correct    47 || Time per epoch:  0.09842
Epoch:    80 || Loss:  1.10346 || Correct    46 || Time per epoch:  0.09778
Epoch:    90 || Loss:  2.20721 || Correct    46 || Time per epoch:  0.09666
Epoch:   100 || Loss:  1.50759 || Correct    49 || Time per epoch:  0.09809
Epoch:   110 || Loss:  1.46235 || Correct    47 || Time per epoch:  0.10250
Epoch:   120 || Loss:  1.18130 || Correct    48 || Time per epoch:  0.09846
Epoch:   130 || Loss:  0.75492 || Correct    48 || Time per epoch:  0.09764
Epoch:   140 || Loss:  1.85645 || Correct    47 || Time per epoch:  0.09763
Epoch:   150 || Loss:  0.62259 || Correct    49 || Time per epoch:  0.09761
Epoch:   160 || Loss:  0.86316 || Correct    47 || Time per epoch:  0.13687
Epoch:   170 || Loss:  0.42979 || Correct    48 || Time per epoch:  0.16689
Epoch:   180 || Loss:  0.95437 || Correct    50 || Time per epoch:  0.09984
Epoch:   190 || Loss:  1.34532 || Correct    47 || Time per epoch:  0.09737
Epoch:   200 || Loss:  1.99121 || Correct    47 || Time per epoch:  0.09764
Epoch:   210 || Loss:  1.53018 || Correct    50 || Time per epoch:  0.09748
Epoch:   220 || Loss:  0.65905 || Correct    48 || Time per epoch:  0.09737
Epoch:   230 || Loss:  1.25711 || Correct    49 || Time per epoch:  0.09765
Epoch:   240 || Loss:  1.03930 || Correct    49 || Time per epoch:  0.09713
Epoch:   250 || Loss:  0.16117 || Correct    50 || Time per epoch:  0.09767
Epoch:   260 || Loss:  0.75199 || Correct    49 || Time per epoch:  0.09775
Epoch:   270 || Loss:  0.90745 || Correct    50 || Time per epoch:  0.09921
Epoch:   280 || Loss:  1.32554 || Correct    50 || Time per epoch:  0.16072
Epoch:   290 || Loss:  1.45179 || Correct    50 || Time per epoch:  0.13271
Epoch:   300 || Loss:  0.36491 || Correct    49 || Time per epoch:  0.09819
Epoch:   310 || Loss:  0.60159 || Correct    50 || Time per epoch:  0.09739
Epoch:   320 || Loss:  1.08264 || Correct    50 || Time per epoch:  0.09821
Epoch:   330 || Loss:  0.14492 || Correct    49 || Time per epoch:  0.09664
Epoch:   340 || Loss:  0.92786 || Correct    50 || Time per epoch:  0.09727
Epoch:   350 || Loss:  0.94650 || Correct    50 || Time per epoch:  0.09722
Epoch:   360 || Loss:  0.73996 || Correct    50 || Time per epoch:  0.09757
Epoch:   370 || Loss:  1.17886 || Correct    47 || Time per epoch:  0.09731
Epoch:   380 || Loss:  0.39339 || Correct    49 || Time per epoch:  0.09760
Epoch:   390 || Loss:  0.55229 || Correct    49 || Time per epoch:  0.12638
Epoch:   400 || Loss:  1.31742 || Correct    49 || Time per epoch:  0.17409
Epoch:   410 || Loss:  1.05392 || Correct    48 || Time per epoch:  0.09765
Epoch:   420 || Loss:  0.86586 || Correct    50 || Time per epoch:  0.09825
Epoch:   430 || Loss:  0.31504 || Correct    50 || Time per epoch:  0.09697
Epoch:   440 || Loss:  0.93252 || Correct    50 || Time per epoch:  0.09739
Epoch:   450 || Loss:  0.50321 || Correct    50 || Time per epoch:  0.10389
Epoch:   460 || Loss:  0.71669 || Correct    50 || Time per epoch:  0.09961
Epoch:   470 || Loss:  0.52318 || Correct    50 || Time per epoch:  0.09718
Epoch:   480 || Loss:  0.44780 || Correct    49 || Time per epoch:  0.09741
Epoch:   490 || Loss:  0.09766 || Correct    50 || Time per epoch:  0.09717
```

Dataset: Split
Backend: GPU
Hidden: 100
LR: 0.05

```
Epoch:     0 || Loss:  7.29856 || Correct    30 || Time per epoch:  5.68361
Epoch:    10 || Loss:  4.96110 || Correct    40 || Time per epoch:  1.88098
Epoch:    20 || Loss:  9.74443 || Correct    25 || Time per epoch:  1.98892
Epoch:    30 || Loss:  4.84711 || Correct    39 || Time per epoch:  1.87550
Epoch:    40 || Loss:  4.40643 || Correct    43 || Time per epoch:  1.93422
Epoch:    50 || Loss:  2.98539 || Correct    44 || Time per epoch:  1.94600
Epoch:    60 || Loss:  4.65592 || Correct    43 || Time per epoch:  1.87556
Epoch:    70 || Loss:  3.23241 || Correct    49 || Time per epoch:  1.93032
Epoch:    80 || Loss:  1.86293 || Correct    47 || Time per epoch:  1.86646
Epoch:    90 || Loss:  2.35521 || Correct    48 || Time per epoch:  1.94626
Epoch:   100 || Loss:  0.99056 || Correct    46 || Time per epoch:  1.87222
Epoch:   110 || Loss:  2.53351 || Correct    48 || Time per epoch:  1.95791
Epoch:   120 || Loss:  2.38217 || Correct    46 || Time per epoch:  1.91514
Epoch:   130 || Loss:  1.68064 || Correct    48 || Time per epoch:  1.90399
Epoch:   140 || Loss:  0.47471 || Correct    49 || Time per epoch:  1.95988
Epoch:   150 || Loss:  1.13329 || Correct    48 || Time per epoch:  1.87276
Epoch:   160 || Loss:  1.08500 || Correct    49 || Time per epoch:  1.95246
Epoch:   170 || Loss:  1.04436 || Correct    49 || Time per epoch:  1.86897
Epoch:   180 || Loss:  1.11847 || Correct    48 || Time per epoch:  1.93738
Epoch:   190 || Loss:  0.76054 || Correct    49 || Time per epoch:  1.91312
Epoch:   200 || Loss:  1.13712 || Correct    49 || Time per epoch:  1.89972
Epoch:   210 || Loss:  0.48208 || Correct    49 || Time per epoch:  1.94485
Epoch:   220 || Loss:  0.53602 || Correct    49 || Time per epoch:  1.86999
Epoch:   230 || Loss:  0.99934 || Correct    49 || Time per epoch:  1.94432
Epoch:   240 || Loss:  1.87460 || Correct    49 || Time per epoch:  1.87323
Epoch:   250 || Loss:  1.02500 || Correct    50 || Time per epoch:  1.94070
Epoch:   260 || Loss:  0.70516 || Correct    49 || Time per epoch:  1.87676
Epoch:   270 || Loss:  1.05425 || Correct    50 || Time per epoch:  1.92227
Epoch:   280 || Loss:  1.46647 || Correct    50 || Time per epoch:  1.94358
Epoch:   290 || Loss:  1.21522 || Correct    50 || Time per epoch:  1.86195
Epoch:   300 || Loss:  0.71102 || Correct    49 || Time per epoch:  1.94752
Epoch:   310 || Loss:  0.79022 || Correct    49 || Time per epoch:  1.87100
Epoch:   320 || Loss:  1.34444 || Correct    49 || Time per epoch:  1.94841
Epoch:   330 || Loss:  1.75439 || Correct    49 || Time per epoch:  1.88745
Epoch:   340 || Loss:  0.69900 || Correct    50 || Time per epoch:  1.93149
Epoch:   350 || Loss:  0.05217 || Correct    49 || Time per epoch:  1.94405
Epoch:   360 || Loss:  0.38922 || Correct    50 || Time per epoch:  1.86621
Epoch:   370 || Loss:  0.87756 || Correct    50 || Time per epoch:  1.94147
Epoch:   380 || Loss:  1.28520 || Correct    49 || Time per epoch:  1.86340
Epoch:   390 || Loss:  1.08639 || Correct    50 || Time per epoch:  1.96440
Epoch:   400 || Loss:  0.38072 || Correct    50 || Time per epoch:  1.87549
Epoch:   410 || Loss:  1.29022 || Correct    49 || Time per epoch:  1.94588
Epoch:   420 || Loss:  0.73392 || Correct    50 || Time per epoch:  1.95597
Epoch:   430 || Loss:  0.64191 || Correct    50 || Time per epoch:  1.88988
Epoch:   440 || Loss:  0.59425 || Correct    50 || Time per epoch:  1.99555
Epoch:   450 || Loss:  1.16145 || Correct    49 || Time per epoch:  1.87641
Epoch:   460 || Loss:  0.18022 || Correct    49 || Time per epoch:  1.96158
Epoch:   470 || Loss:  1.76201 || Correct    49 || Time per epoch:  1.91591
Epoch:   480 || Loss:  0.85149 || Correct    50 || Time per epoch:  1.91267
Epoch:   490 || Loss:  0.42668 || Correct    50 || Time per epoch:  1.97247
```

### XOR

Dataset: XOR
Backend: CPU
Hidden: 100
LR: 0.05

```
Epoch:     0 || Loss:  6.80791 || Correct    28 || Time per epoch:  18.22354
Epoch:    10 || Loss:  4.28319 || Correct    37 || Time per epoch:  0.09793
Epoch:    20 || Loss:  2.93311 || Correct    44 || Time per epoch:  0.09738
Epoch:    30 || Loss:  3.23911 || Correct    44 || Time per epoch:  0.09757
Epoch:    40 || Loss:  2.63329 || Correct    45 || Time per epoch:  0.10524
Epoch:    50 || Loss:  2.51675 || Correct    45 || Time per epoch:  0.17248
Epoch:    60 || Loss:  2.72185 || Correct    45 || Time per epoch:  0.12250
Epoch:    70 || Loss:  1.28917 || Correct    46 || Time per epoch:  0.09658
Epoch:    80 || Loss:  1.48718 || Correct    46 || Time per epoch:  0.09691
Epoch:    90 || Loss:  0.74941 || Correct    46 || Time per epoch:  0.09767
Epoch:   100 || Loss:  2.23547 || Correct    46 || Time per epoch:  0.09663
Epoch:   110 || Loss:  1.35189 || Correct    45 || Time per epoch:  0.09711
Epoch:   120 || Loss:  1.30326 || Correct    46 || Time per epoch:  0.09703
Epoch:   130 || Loss:  3.71375 || Correct    46 || Time per epoch:  0.09878
Epoch:   140 || Loss:  2.33340 || Correct    47 || Time per epoch:  0.10197
Epoch:   150 || Loss:  1.69693 || Correct    47 || Time per epoch:  0.10087
Epoch:   160 || Loss:  2.47193 || Correct    47 || Time per epoch:  0.15403
Epoch:   170 || Loss:  1.62141 || Correct    46 || Time per epoch:  0.15398
Epoch:   180 || Loss:  0.80703 || Correct    47 || Time per epoch:  0.09863
Epoch:   190 || Loss:  2.30713 || Correct    47 || Time per epoch:  0.09952
Epoch:   200 || Loss:  1.80140 || Correct    47 || Time per epoch:  0.09861
Epoch:   210 || Loss:  0.90898 || Correct    47 || Time per epoch:  0.10057
Epoch:   220 || Loss:  2.64474 || Correct    47 || Time per epoch:  0.09898
Epoch:   230 || Loss:  0.48834 || Correct    47 || Time per epoch:  0.10073
Epoch:   240 || Loss:  1.31547 || Correct    48 || Time per epoch:  0.09842
Epoch:   250 || Loss:  2.15667 || Correct    47 || Time per epoch:  0.09809
Epoch:   260 || Loss:  0.81131 || Correct    47 || Time per epoch:  0.09914
Epoch:   270 || Loss:  2.26540 || Correct    48 || Time per epoch:  0.12321
Epoch:   280 || Loss:  0.20556 || Correct    49 || Time per epoch:  0.17621
Epoch:   290 || Loss:  2.12707 || Correct    48 || Time per epoch:  0.09695
Epoch:   300 || Loss:  0.49326 || Correct    47 || Time per epoch:  0.09731
Epoch:   310 || Loss:  1.31325 || Correct    47 || Time per epoch:  0.09725
Epoch:   320 || Loss:  2.58391 || Correct    47 || Time per epoch:  0.09667
Epoch:   330 || Loss:  0.64292 || Correct    47 || Time per epoch:  0.09721
Epoch:   340 || Loss:  0.66021 || Correct    49 || Time per epoch:  0.09732
Epoch:   350 || Loss:  0.74039 || Correct    48 || Time per epoch:  0.09733
Epoch:   360 || Loss:  1.98287 || Correct    48 || Time per epoch:  0.09703
Epoch:   370 || Loss:  0.73640 || Correct    49 || Time per epoch:  0.09780
Epoch:   380 || Loss:  1.76571 || Correct    47 || Time per epoch:  0.09761
Epoch:   390 || Loss:  1.45749 || Correct    49 || Time per epoch:  0.16815
Epoch:   400 || Loss:  1.29377 || Correct    48 || Time per epoch:  0.14178
Epoch:   410 || Loss:  0.69903 || Correct    48 || Time per epoch:  0.10009
Epoch:   420 || Loss:  1.29314 || Correct    49 || Time per epoch:  0.09747
Epoch:   430 || Loss:  1.73983 || Correct    48 || Time per epoch:  0.09669
Epoch:   440 || Loss:  0.28013 || Correct    49 || Time per epoch:  0.09689
Epoch:   450 || Loss:  0.39230 || Correct    49 || Time per epoch:  0.09722
Epoch:   460 || Loss:  0.27196 || Correct    49 || Time per epoch:  0.09699
Epoch:   470 || Loss:  0.83664 || Correct    49 || Time per epoch:  0.09669
Epoch:   480 || Loss:  0.10319 || Correct    49 || Time per epoch:  0.09758
Epoch:   490 || Loss:  0.78115 || Correct    49 || Time per epoch:  0.09673

```

Dataset: XOR
Backend: GPU
Hidden: 100
LR: 0.05

```
Epoch:     0 || Loss:  6.60616 || Correct    29 || Time per epoch:  4.34647
Epoch:    10 || Loss:  7.75637 || Correct    38 || Time per epoch:  1.92764
Epoch:    20 || Loss:  4.08382 || Correct    40 || Time per epoch:  1.93153
Epoch:    30 || Loss:  5.01938 || Correct    40 || Time per epoch:  1.97435
Epoch:    40 || Loss:  2.83069 || Correct    47 || Time per epoch:  1.87607
Epoch:    50 || Loss:  3.72208 || Correct    43 || Time per epoch:  1.95428
Epoch:    60 || Loss:  2.82434 || Correct    45 || Time per epoch:  1.88109
Epoch:    70 || Loss:  1.34516 || Correct    47 || Time per epoch:  1.94866
Epoch:    80 || Loss:  2.56193 || Correct    50 || Time per epoch:  1.95371
Epoch:    90 || Loss:  1.72377 || Correct    46 || Time per epoch:  1.88983
Epoch:   100 || Loss:  1.76068 || Correct    48 || Time per epoch:  1.96065
Epoch:   110 || Loss:  2.22888 || Correct    49 || Time per epoch:  1.89393
Epoch:   120 || Loss:  1.66461 || Correct    49 || Time per epoch:  1.95306
Epoch:   130 || Loss:  1.10404 || Correct    50 || Time per epoch:  1.91725
Epoch:   140 || Loss:  1.11320 || Correct    48 || Time per epoch:  1.92632
Epoch:   150 || Loss:  2.35142 || Correct    48 || Time per epoch:  1.96738
Epoch:   160 || Loss:  0.88490 || Correct    49 || Time per epoch:  1.87565
Epoch:   170 || Loss:  0.67688 || Correct    49 || Time per epoch:  1.95865
Epoch:   180 || Loss:  0.86934 || Correct    49 || Time per epoch:  1.87605
Epoch:   190 || Loss:  0.74325 || Correct    49 || Time per epoch:  1.95826
Epoch:   200 || Loss:  0.92048 || Correct    49 || Time per epoch:  1.95339
Epoch:   210 || Loss:  0.37536 || Correct    49 || Time per epoch:  1.87274
Epoch:   220 || Loss:  0.62137 || Correct    49 || Time per epoch:  1.96424
Epoch:   230 || Loss:  0.17097 || Correct    49 || Time per epoch:  1.87607
Epoch:   240 || Loss:  0.38058 || Correct    49 || Time per epoch:  1.96488
Epoch:   250 || Loss:  0.21541 || Correct    49 || Time per epoch:  1.89750
Epoch:   260 || Loss:  0.83243 || Correct    49 || Time per epoch:  1.95606
Epoch:   270 || Loss:  0.23149 || Correct    49 || Time per epoch:  1.95696
Epoch:   280 || Loss:  0.43091 || Correct    49 || Time per epoch:  1.88882
Epoch:   290 || Loss:  0.34242 || Correct    49 || Time per epoch:  1.96102
Epoch:   300 || Loss:  1.77023 || Correct    49 || Time per epoch:  1.88385
Epoch:   310 || Loss:  1.94001 || Correct    49 || Time per epoch:  1.98119
Epoch:   320 || Loss:  0.46170 || Correct    49 || Time per epoch:  1.98320
Epoch:   330 || Loss:  0.27046 || Correct    49 || Time per epoch:  1.88130
Epoch:   340 || Loss:  0.30852 || Correct    49 || Time per epoch:  1.97567
Epoch:   350 || Loss:  0.10542 || Correct    49 || Time per epoch:  1.87868
Epoch:   360 || Loss:  0.15285 || Correct    49 || Time per epoch:  1.96293
Epoch:   370 || Loss:  0.43318 || Correct    49 || Time per epoch:  1.92663
Epoch:   380 || Loss:  0.31124 || Correct    49 || Time per epoch:  1.90558
Epoch:   390 || Loss:  0.34115 || Correct    49 || Time per epoch:  1.95438
Epoch:   400 || Loss:  0.01727 || Correct    49 || Time per epoch:  1.87678
Epoch:   410 || Loss:  0.83704 || Correct    49 || Time per epoch:  1.96580
Epoch:   420 || Loss:  1.68363 || Correct    49 || Time per epoch:  1.88321
Epoch:   430 || Loss:  0.13303 || Correct    49 || Time per epoch:  1.95556
Epoch:   440 || Loss:  0.07118 || Correct    48 || Time per epoch:  1.95816
Epoch:   450 || Loss:  0.05079 || Correct    47 || Time per epoch:  1.89113
Epoch:   460 || Loss:  1.45303 || Correct    49 || Time per epoch:  1.95944
Epoch:   470 || Loss:  0.85109 || Correct    49 || Time per epoch:  1.89215
Epoch:   480 || Loss:  0.05142 || Correct    49 || Time per epoch:  1.95261
Epoch:   490 || Loss:  0.05718 || Correct    48 || Time per epoch:  1.88079
```

Dataset: XOR
Backend: CPU
Hidden: 500
LR: 0.05

```
Epoch:     0 || Loss:  76.92976 || Correct    32 || Time per epoch:  18.34954
Epoch:    10 || Loss:  1.35926 || Correct    47 || Time per epoch:  0.73230
Epoch:    20 || Loss:  3.21588 || Correct    46 || Time per epoch:  0.84166
Epoch:    30 || Loss:  3.05700 || Correct    49 || Time per epoch:  0.86942
Epoch:    40 || Loss:  3.17658 || Correct    47 || Time per epoch:  0.75742
Epoch:    50 || Loss:  1.94909 || Correct    49 || Time per epoch:  0.82322
Epoch:    60 || Loss:  0.93115 || Correct    49 || Time per epoch:  0.83951
Epoch:    70 || Loss:  0.34140 || Correct    49 || Time per epoch:  0.73307
Epoch:    80 || Loss:  0.66406 || Correct    48 || Time per epoch:  0.83509
Epoch:    90 || Loss:  2.30551 || Correct    50 || Time per epoch:  0.84786
Epoch:   100 || Loss:  3.61139 || Correct    47 || Time per epoch:  0.72473
Epoch:   110 || Loss:  0.83126 || Correct    49 || Time per epoch:  0.83262
Epoch:   120 || Loss:  1.97192 || Correct    50 || Time per epoch:  0.83627
Epoch:   130 || Loss:  0.30938 || Correct    49 || Time per epoch:  0.72335
Epoch:   140 || Loss:  0.52349 || Correct    49 || Time per epoch:  0.84278
Epoch:   150 || Loss:  0.07128 || Correct    49 || Time per epoch:  0.83411
Epoch:   160 || Loss:  0.53294 || Correct    49 || Time per epoch:  0.72746
Epoch:   170 || Loss:  1.30419 || Correct    49 || Time per epoch:  0.83183
Epoch:   180 || Loss:  0.70312 || Correct    49 || Time per epoch:  0.83272
Epoch:   190 || Loss:  1.79597 || Correct    48 || Time per epoch:  0.72660
Epoch:   200 || Loss:  0.90020 || Correct    49 || Time per epoch:  0.83923
Epoch:   210 || Loss:  1.16867 || Correct    49 || Time per epoch:  0.83951
Epoch:   220 || Loss:  1.54156 || Correct    50 || Time per epoch:  0.73978
Epoch:   230 || Loss:  0.25539 || Correct    49 || Time per epoch:  0.83885
Epoch:   240 || Loss:  0.21401 || Correct    49 || Time per epoch:  0.83789
Epoch:   250 || Loss:  0.48800 || Correct    49 || Time per epoch:  0.72728
Epoch:   260 || Loss:  1.88972 || Correct    48 || Time per epoch:  0.83628
Epoch:   270 || Loss:  0.51600 || Correct    49 || Time per epoch:  0.84448
Epoch:   280 || Loss:  0.02321 || Correct    50 || Time per epoch:  0.72758
Epoch:   290 || Loss:  0.15047 || Correct    49 || Time per epoch:  0.84640
Epoch:   300 || Loss:  0.21489 || Correct    49 || Time per epoch:  0.81544
Epoch:   310 || Loss:  0.24901 || Correct    49 || Time per epoch:  0.74679
Epoch:   320 || Loss:  0.43951 || Correct    50 || Time per epoch:  0.83898
Epoch:   330 || Loss:  0.39011 || Correct    50 || Time per epoch:  0.80078
Epoch:   340 || Loss:  0.19627 || Correct    50 || Time per epoch:  0.76696
Epoch:   350 || Loss:  0.11633 || Correct    50 || Time per epoch:  0.83724
Epoch:   360 || Loss:  0.48908 || Correct    49 || Time per epoch:  0.77567
Epoch:   370 || Loss:  0.37118 || Correct    49 || Time per epoch:  0.79994
Epoch:   380 || Loss:  0.19186 || Correct    49 || Time per epoch:  0.84142
Epoch:   390 || Loss:  1.44289 || Correct    50 || Time per epoch:  0.73759
Epoch:   400 || Loss:  1.84813 || Correct    48 || Time per epoch:  0.81961
Epoch:   410 || Loss:  0.00150 || Correct    49 || Time per epoch:  0.82945
Epoch:   420 || Loss:  0.04482 || Correct    49 || Time per epoch:  0.72373
Epoch:   430 || Loss:  1.34708 || Correct    50 || Time per epoch:  0.83246
Epoch:   440 || Loss:  0.39877 || Correct    49 || Time per epoch:  0.84474
Epoch:   450 || Loss:  0.06718 || Correct    49 || Time per epoch:  0.72443
Epoch:   460 || Loss:  0.01587 || Correct    50 || Time per epoch:  0.84396
Epoch:   470 || Loss:  0.82684 || Correct    49 || Time per epoch:  0.83927
Epoch:   480 || Loss:  0.23162 || Correct    50 || Time per epoch:  0.73161
Epoch:   490 || Loss:  0.02331 || Correct    50 || Time per epoch:  0.84179
```

Dataset: XOR
Backend: GPU
Hidden: 500
LR: 0.05

```
Epoch:     0 || Loss:  66.66105 || Correct    26 || Time per epoch:  4.56126
Epoch:    10 || Loss:  1.39066 || Correct    45 || Time per epoch:  2.52134
Epoch:    20 || Loss:  4.20055 || Correct    45 || Time per epoch:  2.54909
Epoch:    30 || Loss:  1.63435 || Correct    44 || Time per epoch:  2.52698
Epoch:    40 || Loss:  1.92903 || Correct    46 || Time per epoch:  2.51618
Epoch:    50 || Loss:  1.86650 || Correct    50 || Time per epoch:  2.53035
Epoch:    60 || Loss:  1.70978 || Correct    48 || Time per epoch:  2.53956
Epoch:    70 || Loss:  0.82648 || Correct    50 || Time per epoch:  2.52403
Epoch:    80 || Loss:  0.49825 || Correct    50 || Time per epoch:  2.51371
Epoch:    90 || Loss:  0.45301 || Correct    50 || Time per epoch:  2.49895
Epoch:   100 || Loss:  0.65924 || Correct    49 || Time per epoch:  2.49445
Epoch:   110 || Loss:  0.85197 || Correct    50 || Time per epoch:  2.49448
Epoch:   120 || Loss:  0.56903 || Correct    50 || Time per epoch:  2.48237
Epoch:   130 || Loss:  0.20003 || Correct    50 || Time per epoch:  2.49669
Epoch:   140 || Loss:  0.25957 || Correct    50 || Time per epoch:  2.48734
Epoch:   150 || Loss:  0.69386 || Correct    50 || Time per epoch:  2.48817
Epoch:   160 || Loss:  0.29471 || Correct    50 || Time per epoch:  2.48345
Epoch:   170 || Loss:  0.41978 || Correct    50 || Time per epoch:  2.50649
Epoch:   180 || Loss:  0.08283 || Correct    50 || Time per epoch:  2.50754
Epoch:   190 || Loss:  0.97569 || Correct    50 || Time per epoch:  2.50714
Epoch:   200 || Loss:  0.13449 || Correct    50 || Time per epoch:  2.49803
Epoch:   210 || Loss:  0.61050 || Correct    50 || Time per epoch:  2.48258
Epoch:   220 || Loss:  0.05742 || Correct    50 || Time per epoch:  2.49355
Epoch:   230 || Loss:  0.16341 || Correct    50 || Time per epoch:  2.50641
Epoch:   240 || Loss:  0.83830 || Correct    50 || Time per epoch:  2.50201
Epoch:   250 || Loss:  0.56408 || Correct    50 || Time per epoch:  2.49401
Epoch:   260 || Loss:  0.58886 || Correct    50 || Time per epoch:  2.49454
Epoch:   270 || Loss:  0.61434 || Correct    50 || Time per epoch:  2.51382
Epoch:   280 || Loss:  0.28423 || Correct    50 || Time per epoch:  2.52634
Epoch:   290 || Loss:  0.75735 || Correct    50 || Time per epoch:  2.53270
Epoch:   300 || Loss:  0.19432 || Correct    50 || Time per epoch:  2.50702
Epoch:   310 || Loss:  0.20030 || Correct    50 || Time per epoch:  2.50111
Epoch:   320 || Loss:  0.08164 || Correct    50 || Time per epoch:  2.49510
Epoch:   330 || Loss:  0.88259 || Correct    49 || Time per epoch:  2.48761
Epoch:   340 || Loss:  0.29611 || Correct    50 || Time per epoch:  2.50932
Epoch:   350 || Loss:  0.02344 || Correct    50 || Time per epoch:  2.50552
Epoch:   360 || Loss:  0.04437 || Correct    50 || Time per epoch:  2.54152
Epoch:   370 || Loss:  0.07200 || Correct    50 || Time per epoch:  2.49936
Epoch:   380 || Loss:  0.25661 || Correct    50 || Time per epoch:  2.49236
Epoch:   390 || Loss:  0.19218 || Correct    50 || Time per epoch:  2.49501
Epoch:   400 || Loss:  0.03950 || Correct    50 || Time per epoch:  2.50482
Epoch:   410 || Loss:  0.12886 || Correct    50 || Time per epoch:  2.49639
Epoch:   420 || Loss:  0.02836 || Correct    50 || Time per epoch:  2.49644
Epoch:   430 || Loss:  0.02420 || Correct    50 || Time per epoch:  2.48882
Epoch:   440 || Loss:  0.01050 || Correct    50 || Time per epoch:  2.50553
Epoch:   450 || Loss:  0.52836 || Correct    50 || Time per epoch:  2.49256
Epoch:   460 || Loss:  0.37681 || Correct    50 || Time per epoch:  2.52336
Epoch:   470 || Loss:  0.08045 || Correct    50 || Time per epoch:  2.54079
Epoch:   480 || Loss:  0.04526 || Correct    50 || Time per epoch:  2.54303
Epoch:   490 || Loss:  0.34197 || Correct    50 || Time per epoch:  2.51295
```