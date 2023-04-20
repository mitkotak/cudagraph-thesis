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
  However, it is not the first choice for deploying production scale
  applications. This is due to a lack of general array fusion capabilities in
  modern array-based frameworks. In this work, we accelarate programs using
  n-d array paradigm by targetting the concurrecy available across the array
  operations in a workload. We do this by extending PYTATO, a
  lazy-evalulation based array package with NUMPY-like semantics, to target
  NVIDIA's CUDA Graph API. While works such as Legate Numpy are also based on
  data flow graphs, it does not contain any scheduling heuristics thus
  restricting its view of the global optimization space. Other works such as
  CuPy, Julia and JAX still rely on a single stream dispatch. While we
  achieve our performance through an application-driven program
  transformation, both PYTORCH and JAX rely on expensive black-box algorihtms
  (Loop / Kernel fusion). To evaluate the soundness of this approach, we port
  a suite of complex operators that represent real world workloads to our
  framework and compare the performance with a version where the array
  operations are executed one after the other. We conclude with some insights
  on NVIDIA's runtime scheduling algorithm using a set of micro-benchmarks to
  propose a roofline-model for task-graph based parallelism on GPUs.

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

  \;

  1. What problems do CUDA Graphs solve ?

  CUDA Graphs, first introduced in CUDA 10, is a new programming model that
  allows asynchronous execution of a user-defined DAG. These DAGs are made up
  of a set of node representing operations such as memory copies and kernel
  launches, connected by edges representing run-after dependencies which are
  defined separetly from its execution through a custom API. In addition to
  releving the programmer from manually partitioning the workload onto
  streams and events, CUDA Graphs also enable a define-once-run-repeatedly
  execution flow by separating the definition and execution of the graph. We
  leverage both of these features in our system by using PYTATO to map array
  operations onto a data flow graph and also caching the code that creates
  the execution graph.

  \;

  2. Why having a CUDAGraph-based array package makes sense ?

  While there have been attempts at simplifying CUDA Graph API usage and
  creating task graph based array-packages, there's a need for an end-to-end
  NUMPY-based package that leverages CUDA Graph's runtime scheduler without
  incurring significant overheads. We attempt to tackle this problem by
  extending PYCUDA to allow calls to the CUDAGraph API and then porting into
  PYTATO's lazy-evaluation based interface.

  \;

  3. Key contributions:

  a. Extend PYCUDA to generate CUDA Graph custom API C driver code.

  b. Map PYTATO's IR onto PyCUDA-CUDA Graph code.

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

  <subsection|CUDA Graphs>

  CUDA Graphs are the most recent innovations to the CUDA driver API. Graphs
  are a step forward compared to the more traditional CUDA streams: a stream
  in CUDA is a queue of copy and compute commands. Within a stream, enqueueud
  operations are implicitly synchronized by the GPU in order to execute them
  in the same order as they are placed into the stream by the programmer.
  Streams allow for aschnronous compute and copy, meaning that CPU cores
  dispatch commands without waiting for their GPU-side completition: even in
  asynchronous submissions, little to no control is left to the programmer
  with respect to when commands are inserted/fetched to/from the stream and
  then dispatched to the GPU engines, with these operations potentially
  overallaping in time.

  CUDA Graphs improve on this approach by allowing the programmer to
  construct a graph of compute, host and copy operations with arbitary intra-
  and inter- stream synchronization, to then dispatch the previously
  displayed operations with a single CPU runtime function. Dispatching a
  CUDAGraph can be an iterative or periodic operation so GPU-CPU taskets can
  be implemented as periodic DAGs.

  \;

  (walk through some CUDAGraph driver code)

  \;

  <subsection|PYCUDA>

  PYCUDA is a run time code generation based PYTHON package that offers a
  high level scripting interface for GPU programming. In this context, since
  PYCUDA already provided a rich ecosystem of abstractions around the CUDA
  driver API, we decided to extend PYCUDA by wrapping the new CUDAGraph
  custom API functions.

  \;

  (Provide example of PyCUDA CUDAGraph code)

  \;

  <subsection|PYTATO>

  PYTATO is a lazy-evaluation programming based PYTHON package that offers a
  subset of NUMPY operations for manipulating multidimensional arrays. This
  provides ease of convenience in managing scientific computing workloads
  (PDE-based numerical methods, deep learning, computational statistics etc.)
  where the higher dimensional vizualization of data is close to the
  mathematical notation.

  In the context of CUDAGraphs, the compilation pipeline is split into the
  following parts:

  Step 1: PYTATO IR that encodes user defined array compuations as a DAG.

  Step 2: PYTATO IR's visitor which in this case is a PyCUDA-CUDAGraph mapper
  to perform term rewriting on built expression.

  Step 3: PyCUDA-CUDAGraph Code Generation

  (Provide a sample of generated code and explain all of the different
  pieces)

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

  \;

  \;

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|font-base-size|11>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|?|1|../../../.TeXmacs/texts/scratch/no_name_2.tm>>
    <associate|auto-2|<tuple|1|2|../../../.TeXmacs/texts/scratch/no_name_2.tm>>
    <associate|auto-3|<tuple|2|3|../../../.TeXmacs/texts/scratch/no_name_2.tm>>
    <associate|auto-4|<tuple|2.1|4|../../../.TeXmacs/texts/scratch/no_name_2.tm>>
    <associate|auto-5|<tuple|2.2|4|../../../.TeXmacs/texts/scratch/no_name_2.tm>>
    <associate|auto-6|<tuple|2.3|4|../../../.TeXmacs/texts/scratch/no_name_2.tm>>
    <associate|auto-7|<tuple|3.3|4|../../../.TeXmacs/texts/scratch/no_name_2.tm>>
    <associate|auto-8|<tuple|4|?|../../../.TeXmacs/texts/scratch/no_name_2.tm>>
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

      2.<space|2spc>RELATED WORK <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>

      3.<space|2spc>OVERVIEW <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>

      <with|par-left|<quote|1tab>|3.1.<space|2spc>CUDA Graphs
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|3.2.<space|2spc>PYCUDA
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|3.3.<space|2spc>PYTATO
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>
    </associate>
  </collection>
</auxiliary>