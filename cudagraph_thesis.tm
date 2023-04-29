<TeXmacs|2.1.1>

<style|<tuple|tmbook|padded-paragraphs|page-numbers>>

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
  While there has been a lot of effort in productionizing n-d array
  applications through kernel and loop fusion, very little attention has been
  paid to leveraging the concurrency across array operations. In this work,
  we target this concurrency by extending <verbatim|Pytato>, a
  lazy-evalulation based array package with <verbatim|Numpy>-like semantics,
  to wrap around NVIDIA's <verbatim|CUDA Graph> API. To evaluate the
  soundness of this approach, we port a suite of complex operators that
  represent real world workloads to our framework and compare the performance
  with a version where the array operations are executed one after the other.
  We conclude with some insights on NVIDIA's runtime scheduling algorithm
  using a set of micro-benchmarks to propose a roofline-model for task-graph
  based parallelism on GPUs.

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

  \;

  \;

  \;

  \;

  \;

  <section|INTRODUCTION>

  <verbatim|CUDA Graph>, first introduced in <verbatim|CUDA> <verbatim|10>,
  is a task-based programming model that allows asynchronous execution of a
  user-defined Directed Acyclic Graph (DAG). These DAGs are made up of a set
  of node representing operations such as memory copies and kernel launches,
  connected by edges representing run-after dependencies which are defined
  separetly from its execution through a custom API. In addition to releving
  the programmer from the burden of dealing with low-level details such as
  prefetching, data transfers, scheduling of tasks, or
  synchronizations,<space|1em><verbatim|CUDA Graphs> also enable a
  define-once-run-repeatedly execution flow by separating the definition and
  execution of the graph. At runtime, the scheduler knows the (i) the state
  of different resources, (oo)

  Array programming is a fundamental computation model that supports a wide
  variety of features, including array slicing and arbitary element-wise,
  reduction and broadcast operators allowing the interface to correspond
  closely to the mathematical needs of the applications. The concurrency
  available across these array operation nodes offers an opportunity to the
  runtime scheduler to saturate all of the available execution units. Our
  system attemps to realize this concurrency by extending
  <samp|<verbatim|Pycuda>> to allow calls to the <verbatim|CUDA Graph> API
  and then mapping the array operations onto a DAG through
  <verbatim|Pytato'>s IR to generate <verbatim|Pycuda>-<verbatim|CUDA Graph>
  code.

  The roof-line performance of any task based application is limited by its
  task types and dependencies among tasks. We attempt to provide some
  theoretical performance bounds by applying the HEFT heuristic onto our
  system.

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

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  <section|OVERVIEW>

  <subsection|<verbatim|CUDA Graphs>>

  <verbatim|CUDA Graph>s offer a graph-based alternative to the more
  traditional <verbatim|CUDA> streams: a stream in <verbatim|CUDA> is a queue
  of copy and compute commands. Within a stream, enqueueud operations are
  implicitly synchronized by the GPU in order to execute them in the same
  order as they are placed into the stream by the programmer. Streams allow
  for aschnronous compute and copy, meaning that CPU cores dispatch commands
  without waiting for their GPU-side completition: even in asynchronous
  submissions, little to no control is left to the programmer with respect to
  when commands are inserted/fetched to/from the stream and then dispatched
  to the GPU engines, with these operations potentially overallaping in time.

  <verbatim|CUDA Graphs> improve on this approach by allowing the programmer
  to construct a graph of compute, host and copy operations with arbitary
  intra- and inter- stream synchronization, to then dispatch the previously
  displayed operations with a single CPU runtime function. Dispatching a
  <verbatim|CUDA Graph> can be an iterative or periodic operation so GPU-CPU
  taskets can be implemented as periodic DAGs.

  We wrapped the <verbatim|CUDA Graph> driver API using <verbatim|Pycuda>
  which provided a high level <verbatim|Python> scripting interface for GPU
  programming.

  \;

  (Provide example of PyCUDA CUDAGraph code)

  <subsection|<verbatim|Loopy>>

  We rely to <verbatim|Loopy> which is <verbatim|Python>-based transformation
  toolkit to generate transformed kernels which are then passed onto
  <verbatim|Pycuda's> run time code generation interface. We make use of the
  following components in our pipeline:

  1. <em|Loop Domains>: \ The upper and lower bounds of the result array's
  memory access pattern in the <verbatim|OpenCL> format sourced from the
  <verbatim|shape> attribute within <verbatim|IndexLambda> and expressed
  using the <verbatim|isl> library.

  (Example)

  2. <em|Statement:> A set of instructions specificed in conjuction with an
  iteration domain which encodes an assignment to an entry of an array. The
  right-hand side of an assignment consists of an expression that may consist
  of arithmetic operations and calls to functions. \ 

  (Example)

  3. <em|Kernel Data>: A sorted list of arguments capturing all of the array
  node's dependencies

  (Example)<htab|5mm>.

  <subsection|<verbatim|Pytato>>

  <verbatim|Pytato> is a lazy-evaluation programming based <verbatim|Python>
  package that offers a subset of <verbatim|Numpy> operations for
  manipulating multidimensional arrays. This provides ease of convenience in
  managing scientific computing workloads (PDE-based numerical methods, deep
  learning, computational statistics etc.) where the higher dimensional
  vizualization of data is close to the mathematical notation.

  In the context of <verbatim|CUDAGraphs>, the compilation pipeline is split
  into the following parts:

  Step 1: <verbatim|Pytato> IR that encodes user defined array compuations as
  a <verbatim|DAG>.

  Step 2: <verbatim|Pytato> IR's visitor which in this case is a
  <verbatim|Pycuda-CudaGraph> mapper onto <verbatim|Pytato's> canonical
  representation <verbatim|IndexLambda>.

  Step 3:<verbatim| Pycuda-CudaGraph> Code Generation

  <\named-algorithm|: <strong|Pseudo-code for DAG traversal in
  <verbatim|Pytato>>>
    <strong|function> <name|AddPycudaCudagraphNode>(depNode)

    <space|1em><with|color|dark cyan|<em|{Returns the <verbatim|Loopy> kernel
    corresponding to an array operation}><with|color|dark cyan|>>

    <space|1em><text-dots>

    <strong|end function>

    \;

    <strong|function> <name|VisitDependencies>(Node)<space|4em>

    <space|1em>while Node.dependencies <math|\<neq\>> <math|\<phi\>>

    <space|2em>for depNode in Node.dependencies

    <space|3em>if depNode.visited = False

    <space|4em><name|VisitDependencies>(depNode)

    <space|4em>depNode.visited = True

    <space|1em><name|AddPycudaCudagraphNode>(depNode)

    <strong|end function>

    \;

    <strong|function> <name|TraverseDAG>(graph)

    <space|1em>graphSort = <name|TopologicalSort>(graph)

    <space|1em><name|TransformNodestoIndexLambda>(graphSort)

    <space|1em><name|VisitDependencies>(graphSort.resultNode)

    <space|1em>

    \ 
  </named-algorithm>

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  <section|Related work>

  <verbatim|StarPU> is a runtime system developed by Inria Bordeaux, France
  specifically designed for heterogenous multicore architectures. Just like
  <verbatim|CudaGraphs>, <verbatim|StarPU >supports a task-based programming
  model by scheduling tasks efficiently using well-known generic dynamic and
  task graph scheduling policies from the literature, and optimizing data
  transfers using prefetching and overallaping, in particular. Complete
  description of StarPU can be found in the work by Augonnet. The StarPU
  scheduling system also offers a large set of features which include full
  control over the scheduling policy, support for hybrid platforms and
  efficient handling of data transfers. However, unfortunately StarPU does
  not yet offer a high level <verbatim|Numpy-like> interface that exposes its
  powerful API.\ 

  <verbatim|Legion> is a data-centric programming model and runtime system
  for achieving high performance on distributed heterogenous architectures
  developed at Stanford University. It provides a <verbatim|Numpy-like>
  interface that allows programmers to explicitly declare different
  properties of program data, such as data organization, partioning and
  control the mapping of tasks onto different architectures. The programming
  model uses a software out-of-order processor, or SOOP, for scheduling tasks
  which takes locality and independence properties captured by logical
  regions while making scheduling decisions. While it provides the ability to
  partition data in multiple ways and to migrate data dynamically between
  these views as the application moves between different phases of
  computation, it restricts its view of the global optimization space.

  (Still going through array based frameworks)

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
    <associate|auto-4|<tuple|2.1|2>>
    <associate|auto-5|<tuple|2.2|2>>
    <associate|auto-6|<tuple|2.3|3>>
    <associate|auto-7|<tuple|3|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|font-shape|<quote|small-caps>|Abstract>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <pageref|auto-1><vspace|0.5fn>

      1.<space|2spc>INTRODUCTION <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>

      2.<space|2spc>OVERVIEW <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>

      <with|par-left|<quote|1tab>|2.1.<space|2spc>CUDA Graphs
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|2.2.<space|2spc>PYTATO
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      3.<space|2spc>Related work <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>
    </associate>
  </collection>
</auxiliary>