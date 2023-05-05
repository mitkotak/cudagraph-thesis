<TeXmacs|2.1.1>

<style|<tuple|tmbook|padded-paragraphs|page-numbers|python>>

<\body>
  \;

  <space|10em><strong|<strong|<strong|<space|5em><abstract-data|<abstract|>>>>>

  \;

  \;

  \;

  \;

  \;

  Array programming paradigm offers routines to express the computation
  cleanly for for a wide variety of scientific computing applications (Finite
  Element Method, Stencil Codes, Image Processing, Machine Learning, etc.).
  These routines are often implemented in interpreted languages like Python
  which are too unconstrained for performance tuning. While there have been a
  lot of efforts in scaling up n-d array applications through kernel and loop
  fusion, very little attention has been paid towards harnesssing the
  concurrency across array operations. The dependency pattern between these
  array operations allow multiple array operations to be executed
  concurrently. This concurrency can be targeted to accelarate the
  application's performance. NVIDIA's CUDAGraph API offers a task programming
  model that can help realise this concurrency by overcoming kernel launch
  latencies and exploiting kernel overlap by scheduling multiple kernel
  executions . In this work we create a task-based lazy-evaluation array
  programming interface by mapping array operations onto CUDAGraphs using
  <verbatim|Pytato's> IR and <verbatim|PyCUDA's> GPU scripting interface. To
  evaluate the soundness of this approach, we port a suite of complex
  operators that represent real world workloads to our framework and compare
  the performance with a version where the array operations are executed one
  after the other. We conclude with some insights on NVIDIA's runtime
  scheduling algorithm using a set of micro-benchmarks to motivate future
  performance modelling work.

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  <compound|section|INTRODUCTION>

  Array programming is a fundamental computation model that supports a wide
  variety of features, including array slicing and arbitary element-wise,
  reduction and broadcast operators allowing the interface to correspond
  closely to the mathematical needs of the applications. <verbatim|PyCUDA>
  and several other array-based frameworks serve as drop-in replacements for
  accelarating <verbatim|Numpy-like> operations on GPUs. While abstractions
  like <verbatim|GPUArray>'s offer a very convenient abstraction for edoing
  \Pstream\Q computing on these arrays, they are not yet able to
  automatically schedule and manage overlapping array operations onto
  multiple streams. The concurrency available in the dependency pattern for
  these array routines can be exploited to saturate all of the available
  execution units.

  Currently the only way to tap into this concurrency is by manually
  scheduling array operations onto mutliple CUDA streams which typically
  requires a lot of experimentation since information about demand resources
  of a kernel such as GPU threads, registers and shared memory is only
  accessible at runtime.

  \;

  (Figure with knl_where python code + Stream/Graph cartoon + overlap picture
  )

  \;

  \;

  Our system automatically realises this concurrency across array operations
  using <verbatim|NVIDIA's> <verbatim|CUDAGraph> API. <verbatim|CUDAGraph>,
  first introduced in <verbatim|CUDA> <verbatim|10>, is a task-based
  programming model that allows asynchronous execution of a user-defined
  Directed Acyclic Graph (DAG). These DAGs are made up of a set of node
  representing operations such as memory copies and kernel launches,
  connected by edges representing run-after dependencies which are defined
  separetly from its execution through a custom API.

  \;

  We implement this system by:

  1. Extending <samp|<verbatim|PyCUDA>> to allow calls to the
  <verbatim|CUDAGraph> API\ 

  2. Mapping the array operations onto a DAG through <verbatim|Pytato'>s IR
  to generate <verbatim|PyCUDA>- <verbatim|CUDAGraph> code.

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  <compound|section|OVERVIEW>

  <subsection|<verbatim|CUDA Graphs>>

  \;

  CUDAGraphs provide a way to execute a partially ordered set of
  compute/memory operations on a GPU, compared to the fully ordered
  <verbatim|CUDA> streams: : a stream in <verbatim|CUDA> is a queue of copy
  and compute commands. Within a stream, enqueueud operations are implicitly
  synchronized by the GPU in order to execute them in the same order as they
  are placed into the stream by the programmer. Streams allow for aschnronous
  compute and copy, meaning that CPU cores dispatch commands without waiting
  for their GPU-side completition: even in asynchronous submissions, little
  to no control is left to the programmer with respect to when commands are
  inserted/fetched to/from the stream and then dispatched to the GPU engines,
  with these operations potentially overallaping in time.

  CUDAGraphs faciliate the mapping of independent A CUDAGraph is a set of
  nodes representing memory/compute operations, connected by edges
  representing run-after dependencies. CUDA 10 introduces explicit APIs for
  creating graphs, e.g. cuGraphCreate, to create a graph;
  cuGraphAddMemAllocNode/cuGraphAddKernelNode/cuGraphMemFreeNode, to add a
  new node to the graph with the corresponding run-after dependencies with
  previous nodes to be exected on the GPU; cuGraphInstantiate, to create an
  executable graph in a stream; and a cuGraphLaunch, to launch an executable
  graph. We wrapped this API using <verbatim|PyCUDA> which provided a high
  level <verbatim|Python> scripting interface for GPU programming. The table
  below lists commonly used PyCUDA-CUDAGraph functions. Refere to [link] for
  a comprehensive list of wrapped functions.

  \;

  <tabular*|<tformat|<cwith|1|-1|1|-1|cell-valign|c>|<twith|table-width|1par>|<twith|table-hmode|exact>|<cwith|1|1|1|-1|cell-hyphen|t>|<cwith|1|1|1|-1|cell-tborder|1ln>|<cwith|11|11|1|-1|cell-bborder|1ln>|<cwith|1|-1|1|1|cell-lborder|0ln>|<cwith|1|-1|2|2|cell-rborder|0ln>|<table|<row|<\cell>
    Operations
  </cell>|<\cell>
    <verbatim|PyCUDA> routines
  </cell>>|<row|<cell|Memory Allocation>|<verbatim|add_memalloc_node>>|<row|<cell|Kernel
  Execution>|<cell|<verbatim|add_kernel_node>>>|<row|<cell|Host to Device
  Copy>|<cell|<verbatim|add_memcpy_htod_node>>>|<row|<cell|Device to Device
  Copy>|<cell|<verbatim|add_memcpy_dtod_node>>>|<row|<cell|Device to Host
  Copy>|<cell|<verbatim|add_memcpy_dtoh_node>>>|<row|<cell|Memory
  Free>|<cell|<verbatim|add_memfree_node>>>|<row|<cell|Graph
  Creation>|<cell|<verbatim|Graph>>>|<row|<cell|Graph
  Instantiation>|<cell|<verbatim|GraphExec>>>|<row|<cell|Update ExecGraph
  arguments>|<cell|<verbatim|batched_set_kernel_node_arguments>>>|<row|<cell|Graph
  Launch>|<cell|<verbatim|launch>>>>>>

  \;

  \;

  Here's a simple example demonstrating the CUDAGraph functionality:

  <\enumerate-alpha>
    <compound|item>Create a <verbatim|Graph> through <em|cuGraphCreate>.
    Define and load the kernel function using
    <em|cuModuleLoadData>/<em|cuModuleGetFunction> calls baked into
    <verbatim|SourceModule> abstraction.

    <item>Create and allocate memory for Numpy arrays. Transfer the memory to
    GPU via <em|cuGraphAddMemcpyNode>.

    <item>Add a kernel node with the <verbatim|memcpy_htod_node> as
    dependency using <em|cuGraphAddKernelNode>.

    <item>Transfer the memory back to host via <em|cuGraphAddMemcpyNode> with
    <verbatim|kernel node> and <verbatim|memcpy_dtoh_node> as dependencies.

    <item>Instantiate the graph through <em|cuGraphInstantiate> and execute
    it through <em|cuGraphLaunch>.
  </enumerate-alpha>

  <\python-code>
    #!/usr/bin/env python

    \;

    g = drv.Graph() # Create Graph

    \;

    mod = SourceModule("

    \ \ \ \ \ \ #define bIdx(N) ((int) blockIdx.N)\\n#define tIdx(N) ((int)

    \ \ \ \ \ \ threadIdx.N)\\n\\nextern "C" __global__ void
    __launch_bounds__(16) \ \ \ \ \ \ \ \ doublify(double\ 

    \ \ \ \ \ \ *__restrict__ out, double const *__restrict__ _in1)\\n{\\n
    \ {\\n \ \ \ \ \ \ \ \ \ \ \ int const ibatch = 0;\\n\\n \ \ \ out[4 *
    (tIdx(x) / 4) + tIdx(x) + -4 * \ \ \ \ \ (tIdx(x) / 4)] = 2.0 * _in1[4 *
    (tIdx(x) / 4) + tIdx(x) + -4 * ( \ \ \ \ \ \ \ \ \ tIdx(x) / 4)];\\n
    \ }\\n}")

    doublify = mod.get_function("doublify") \ \ \ \ \ # Get kernel function

    \;

    a = np.random.randn(4, 4).astype(np.float64) # Random input array

    a_doubled = np.empty_like(a) \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ # Empty
    Result Array

    a_gpu = drv.mem_alloc(a.nbytes) \ \ \ \ \ \ \ \ \ \ \ \ \ # Allocating
    input memory

    \;

    memcpy_htod_node = g.add_memcpy_htod_node(a_gpu, a, a.nbytes) # HtoD

    \;

    kernel_node = g.add_kernel_node(a_gpu, func=doublify, block=(4, 4, 1),
    \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ dependencies=[memcpy_htod_node])
    #Kernel

    \;

    memcpy_dtoh_node = g.add_memcpy_dtoh_node(a_doubled, a_gpu, a.nbytes,
    \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ [kernel_node,
    memcpy_htod_node]) #DtoH

    \;

    g_exec = drv.GraphExec(g) \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ #
    Instantiate Graph

    g_exec.launch() \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ #
    Execute Graph

    \ 

    \;
  </python-code>

  <subsection|<verbatim|Loopy>>

  Loopy is <verbatim|a Python>-based transformation toolkit to generate
  transformed kernels. We make use of the following components in our
  pipeline to generate performance tuned <verbatim|CUDA> kernels:

  1. <em|Loop Domains>: \ The upper and lower bounds of the result array's
  memory access pattern in the <verbatim|OpenCL> format sourced from the
  <verbatim|shape> attribute within <verbatim|IndexLambda> and expressed
  using the <verbatim|isl> library.

  2. <em|Statement:> A set of instructions specificed in conjuction with an
  iteration domain which encodes an assignment to an entry of an array. The
  right-hand side of an assignment consists of an expression that may consist
  of arithmetic operations and calls to functions. \ 

  3. <em|Kernel Data>: A sorted list of arguments capturing all of the array
  node's dependencies.

  <\python-code>
    lp.make_kernel(

    \ \ domains = "{[_0]:0\<less\>=_0\<less\>4}}",

    \ \ instructions = "out[_0]=2*a[_0]",

    \ \ kernel_data = [lp.GlobalArg("out", shape=lp.auto, dtype="float64"),

    \ \ \ \ lp.GlobalArg("a", shape=lp.auto, dtype="float64")])
  </python-code>

  <subsection|<verbatim|Pytato>>

  <verbatim|Pytato> is a lazy-evaluation programming based <verbatim|Python>
  package that offers a subset of <verbatim|Numpy> operations for
  manipulating multidimensional arrays. This provides ease of convenience in
  managing scientific computing workloads (PDE-based numerical methods, deep
  learning, computational statistics etc.) where the higher dimensional
  vizualization of data is close to the mathematical notation.

  We make use of the following <verbatim|Pytato> nodes in our pipeline:

  1. <em|Placeholder>: A named placeholder for an array whose concrete value
  is supplied during runtime.

  2. <em|IndexLambda>: A canonical representation for capturing array
  operations where operations are supplied as <verbatim|Pymbolic> expressions
  and operands are stored inside a dictionary which maps strings that are
  valid Python identifiers onto <verbatim|PlaceHolde>r objects.

  Here's an example capturing the <verbatim|PyCUDA-CUDAGrap>h workflow shown
  above.

  <\python-code>
    #!/usr/bin/env python

    \;

    import pytato as pt

    import numpy as np

    x = pt.make_placeholder(name="x", shape=(4,4), dtype="float64")

    result = pt.make_dict_of_named_arrays({"2x": 2*x})

    \;

    # {{{ execute

    \;

    import pyopencl as cl

    ctx = cl.create_some_context()

    queue = cl.CommandQueue(ctx)

    prg = pt.generate_loopy(result, cl_device=queue.device)

    a = np.random.randn(4, 4).astype(np.float64)

    _, out = prg(queue, x=x)

    \;

    # }}}
  </python-code>

  \;

  \;

  \;

  \;

  <compound|section|Array Operations to CudaGraph Transformation>

  \;

  In the context of <verbatim|CUDAGraphs>, the compilation pipeline is split
  into the following parts:

  1) <verbatim|Pytato> IR that encodes user defined array compuations as a
  <verbatim|DAG> where nodes correspond to array operations and edges
  represnting dependencies between inputs/outputs of these operations.\ 

  2) <verbatim|Pytato> IR's visitor which in this case is a
  <verbatim|PyCUDA-CUDAGraph> mapper onto <verbatim|Pytato's> canonical
  representation <verbatim|IndexLambda>.

  3)<verbatim| PyCUDA-CUDAGraph> Code Generation

  \;

  <\algorithm>
    Step 1: Capture user-defined DAG through Pytato's array interface.

    \;

    Step 2: Run a topological sort on the graph to facilitate optimal FLOP
    choice and store the graph as a <verbatim|DictOfNamedArrays>. Create a
    corresponding <verbatim|CUDAGraph> object.

    \;

    Step 3:

    \;

    <strong|for> a <math|\<epsilon\>> Nodes in <verbatim|DictOfNamedArrays>
    <strong|do>

    \;

    <space|4em><tabular*|<tformat|<table|<row|<cell|Source Node
    Type>|<cell|Target Node Type>>|<row|<cell|<verbatim|DataWrapper>>|<cell|<verbatim|PlaceHolder>>>|<row|<cell|<verbatim|Roll>>|<cell|<verbatim|IndexLambda>>>|<row|<cell|<verbatim|AxisPermutation>>|<cell|<verbatim|IndexLambda>>>|<row|<cell|<verbatim|IndexBase>>|<cell|<verbatim|IndexLambda>>>|<row|<cell|<verbatim|Reshape>>|<cell|<verbatim|IndexLambda>>>|<row|<cell|<verbatim|Concatenate>>|<cell|<verbatim|IndexLambda>>>|<row|<cell|<verbatim|Einsum>>|<cell|<verbatim|IndexLambda>>>>>>

    <space|3em>

    <\strong>
      done

      \;

      \;
    </strong>

    Step 4:

    \;

    <strong|for> a <math|\<epsilon\>> Nodes in <verbatim|DictOfNamedArrays>
    <strong|do>

    \;

    <space|2em>if a == <verbatim|PlaceHolde>r:

    <\itemize-arrow>
      <item>Link to user provided buffers or generate new buffers via
      <verbatim|GPUArrays>.

      <item>Emit <verbatim|CgenMapperAccumulator> node tracking the allocated
      buffer and dependencies.

      \;
    </itemize-arrow>

    \;

    else if a == <verbatim|IndexLambda>:

    <\itemize-arrow>
      <item>Generate kernel string and launch dimensions by plugging
      <verbatim|IndexLambda> expression into <verbatim|lp.make_kernel>.

      <item>Add kernel node with temporary buffer arguments and corresponding
      result memalloc node with dependencies from child
      <verbatim|CgenMapperAccumulator> nodes going into the
      <verbatim|dependencies> field.

      <item>Emit <verbatim|CgenMapperAccumulator> node tracking the result
      buffer and dependencies.
    </itemize-arrow>

    <strong|done>

    \;

    Step 5: Instantiate and cache the executable graph.

    \;

    Step 6: For every subsequent graph launch the input buffers get updated

    \;

    <strong|for> a <math|\<epsilon\>> Nodes in <verbatim|PlaceHolders>
    <strong|do><space|1em> <htab|5mm>

    <\itemize-arrow>
      <compound|item>Replace kernel node temporary buffers with buffers from
      corresponding <verbatim|CgenMapperAccumulator> nodes using
      <verbatim|batched_set_kernel_node_arguments>
    </itemize-arrow>

    <strong|done>

    \;
  </algorithm>

  The generated code can be split into the following parts:

  1) <em|Kernel Creation>: Load <verbatim|cuModules> using kernel strings
  derived from <verbatim|Loopy.>\ 

  2) <em|CUDAGraph Building>: Build and cache execution graph by traversing
  user-defined DAG.

  3) <em|Memory Allocation:> Update execution graph with allocated/linked
  buffers and launch the graph.

  \;

  <\python-code>
    #!/usr/bin/env python

    \;

    _pt_mod_0 = _pt_SourceModule("

    #define bIdx(N) ((int) blockIdx.N)\\n#define tIdx(N) ((int)

    threadIdx.N)\\n\\nextern "C" __global__ void __launch_bounds__(16)
    doublify(double\ 

    *__restrict__ out, double const *__restrict__ _in1)\\n{\\n \ {\\n
    \ \ \ int const ibatch = 0;\\n\\n \ \ \ out[4 * (tIdx(x) / 4) + tIdx(x) +
    -4 * (tIdx(x) / 4)] = 2.0 * _in1[4 * (tIdx(x) / 4) + tIdx(x) + -4 *
    (tIdx(x) / 4)];\\n \ }\\n}")

    \;

    @cache

    def exec_graph_builder():

    \ \ \ \ _pt_g = _pt_drv.Graph()

    \ \ \ \ _pt_buffer_acc = {}

    \ \ \ \ _pt_node_acc = {}

    \ \ \ \ _pt_memalloc, _pt_array = _pt_g.add_memalloc_node(size=128,
    dependencies=[])

    \ \ \ \ _pt_kernel_0 = _pt_g.add_kernel_node(_pt_array, 139712027164672,
    func=_pt_mod_0.get_function('doublify'), block=(16, 1, 1), grid=(1, 1,
    1), dependencies=[_pt_memalloc])

    \ \ \ \ _pt_buffer_acc['_pt_array'] = _pt_array

    \ \ \ \ _pt_node_acc['_pt_kernel_0'] = _pt_kernel_0

    \ \ \ \ _pt_g.add_memfree_node(_pt_array, [_pt_kernel_0])

    \ \ \ \ return (_pt_g.get_exec_graph(), _pt_g, _pt_node_acc,
    _pt_buffer_acc)

    \;

    def _pt_kernel(allocator=cuda_allocator, dev=cuda_dev, *, _pt_data):

    \ \ \ \ _pt_result = _pt_gpuarray.GPUArray((4, 4), dtype='float64',
    allocator=allocator, dev=dev)

    \ \ \ \ _pt_exec_g, _pt_g, _pt_node_acc, _pt_buffer_acc =
    exec_graph_builder()

    \ \ \ \ _pt_exec_g.batched_set_kernel_node_arguments({_pt_node_acc['_pt_kernel_0']:
    _pt_drv.KernelNodeParams(args=[_pt_result.gpudata, _pt_data.gpudata])})

    \ \ \ \ _pt_exec_g.launch()

    \ \ \ \ _pt_tmp = {'2a': _pt_result}

    \ \ \ \ return _pt_tmp
  </python-code>

  <section|Related work>

  The literature on task-based array programming can be classified roughly
  according to their choice of task granularity.

  <em|Function>: Castro et give an overview of the current task-based
  <verbatim|Python> computing landscape by mentioning several libraries that
  rely on <em|decorators>. A decorator is an instruction set before the
  definition of a function. The decorator function transforms the user
  function (if applicable) into a parallelization-friendly version. Libraries
  such as <verbatim|PyCOMPs>, <verbatim|Pygion>, <verbatim|PyKoKKos> and
  <verbatim|Legion> make use of this core principle to accelarate
  <em|vanilla> <verbatim|Python> code. <verbatim|PyCOMPs> and
  <verbatim|Pygion> both rely on <verbatim|@task> decorator to build a task
  dependency graph and define the order of execution. <verbatim|PyKoKKos>
  ports into the <verbatim|KoKKos> API and passes the <verbatim|@pk.workunit>
  decorator into the <verbatim|parallel_for()> function. <verbatim|Legion>
  uses a data-centric programming model which relies on <em|software
  out-of-order processor> (SOOP), for scheduling tasks which takes locality
  and independence properties captured by logical regions while making
  scheduling decisions.

  In <verbatim|Jug>, arguments take values or outputs of another tasks and
  parallelization is achieved by running more than one <verbatim|Jug>
  processes for distributing the tasks. In <verbatim|Pydron>, decorated
  functions are first translated into an intermediate representation and then
  analyzed by a scheduler which updates the execution graph as each task is
  finished.

  Since all of these frameworks rely on explicit taks declarations, they are
  not able to realise the concurrency available across array operations.

  <em|Stream>: <verbatim|CuPy> serves as a drop-in replacement to
  <verbatim|Numpy> and uses NVIDIA's in-house CUDA frameworks such
  as<verbatim| cuBLAS>, <verbatim|cuDNN> and <verbatim|cuSPARSE> to
  accelerate its performance. <verbatim|Julia> GPU programming models
  use<verbatim| CUDA.jl> to provide a high level mechanics to define
  multidimenstional arrays (<verbatim|CUArray>). Both <verbatim|CuPy> and
  <verbatim|Julia >offer interfaces for <em|implcit> graph construction which
  <em|captures> a <verbatim|CUDAGraph> using existing stream-based APIs.
  Implicit <verbatim|CUDAGraph> construction is more flexible and general,
  but requires to wrangle with conconcurrency details through events and
  streams.

  \;

  \ 

  \;

  \;

  <em|Graph>: <compound|verbatim|JAX> optimizes GPU peformance by translating
  <em|high-level traces> into XL HLO and then performing
  vectorization/parallelization, automatic differentiation, and <verbatim|JIT
  >compilation. Deep learning symbolic mathematical libraries such as
  <verbatim|TensorFlow> and <verbatim|Pytorch> allow neural networks to be
  specified as DAGs along which data is transformed. Just like
  <verbatim|CUDAGraphs>, in <verbatim|TensorFlow>, computational DAGs are
  defined statically so that their compilation and execution yield maximum
  performance. <verbatim|PyTorch> on the other hand offers more control at
  runtime by allowing the modification of executing nodes facilitating the
  implementation of sophosticated training routines.

  <em|Kernel>: <verbatim|StarPU> supports a task-based programming model by
  scheduling tasks efficiently using well-known generic dynamic and task
  graph scheduling policies from the literature, and optimizing data
  transfers using prefetching and overallaping. Each StarPU task describes
  the computation kernel, possible implementations on different architectures
  (CPUs/GPUs), what data is being accessed and how its accessed during
  comptuation (read/write mode). Task dependencies are inferred from data
  dependencies.

  \;

  <compound|section|RESULTS>

  (Plots using the latest version dg_benchmarks)

  (Show speedup by changing height and width for different kernels)

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|font|roman>
    <associate|font-base-size|11>
    <associate|font-family|rm>
    <associate|math-font|roman>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|?|1>>
    <associate|auto-2|<tuple|1|2>>
    <associate|auto-3|<tuple|2|2>>
    <associate|auto-4|<tuple|2.1|3>>
    <associate|auto-5|<tuple|2.2|3>>
    <associate|auto-6|<tuple|2.3|4>>
    <associate|auto-7|<tuple|3|4>>
    <associate|auto-8|<tuple|4|6>>
    <associate|auto-9|<tuple|5|7>>
    <associate|docs-internal-guid-e523d6ab-7fff-3f04-e0e8-c225a0b87000|<tuple|?|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|figure>
      <tuple|normal|<\surround|<hidden-binding|<tuple>|1>|>
        \;
      </surround>|<pageref|auto-3>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|font-shape|<quote|small-caps>|Abstract>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <pageref|auto-1><vspace|0.5fn>

      1.<space|2spc>INTRODUCTION <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>

      2.<space|2spc>OVERVIEW <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>

      <with|par-left|<quote|1tab>|2.1.<space|2spc><with|font-family|<quote|tt>|language|<quote|verbatim>|CUDA
      Graphs> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|2.2.<space|2spc><with|font-family|<quote|tt>|language|<quote|verbatim>|Loopy>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|2.3.<space|2spc><with|font-family|<quote|tt>|language|<quote|verbatim>|Pytato>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      3.<space|2spc>Related work <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>

      4.<space|2spc>RESULTS <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>
    </associate>
  </collection>
</auxiliary>