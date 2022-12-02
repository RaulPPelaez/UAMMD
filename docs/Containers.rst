Containers
==========

Many times a particular module requires some kind of intermediate or temporal storage that is not required between subsequent calls to the module. In this instance one would normally have to choose between two options:
 1. Allocate and deallocate a container at each call: Allocating/Deallocating memory are in principle costly operations. Although :ref:`System`'s allocators can help with that, both thrust and the std library containers come with a little initialization quirk that makes this option many times inconvenient. In particular, the standard for containers requires `default construction of new elements at resizing <https://en.cppreference.com/w/cpp/container/vector/reside/>`_ even when the new elements require no special initialization (like zeroing). This can be circumvented in :cpp:`std::vector`, but :cpp:`thrust::device_container` launches a kernel (at the default CUDA stream) which cannot be avoided when resizing/creating a new vector. Even when the time for launching this kernel and the cost of the kernel itself might be neglegible, the fact that this kernel launches at the default stream makes all other asynchronous GPU operations suffer an implicit synchronization barrier.
 2. Store the intermediate memory persistently: For instance, allocating a temporal storage vector at module creation and only deallocating at destruction. While this provides optimal performance it wastes memory that could be leveraged by other modules in a similar situation.

.. hint::

   Grid-based :ref:`BDHI` modules are examples of modules requiring an intermediate memory that does not need to be persistent. While requiring no memory of previous steps, they typically need to compute the state of a discretized "fluid" as part of their computation.

:ref:`System` provides a common pool cached allocator (see :cpp:any:`System::allocator`) that virtually makes allocation and deallocation of GPU memory free. It does not help, however, with fixing the issue of the unavoidable kernel launch for default-initialization of the memory. For that matter UAMMD exposes the following container:

Uninitialized cached vector
+++++++++++++++++++++++++++


.. cpp:class:: template<class T> uammd::uninitialized_cached_vector<T>

   This container mimics a :cpp:`thrust::device_vector`, but does not default-initialize new elements at creation/resizing. Furthermore, it defaults to using :cpp:any:`System::allocator` for allocation. This makes creation and resizing of a new vector potentially cost-free.
   

.. hint:: This container can be found in the :code:`utils/container.h` source.

Managed allocator
++++++++++++++++++

UAMMD also exposes a managed memory device vector. It is simply a :code:`thrust::device_vector` paired with a managed memory allocator.

.. cpp:class:: template<class T> uammd::managed_vector<T>

   A device container that allocates managed memory (pointers to managed memory are accessible from CPU and GPU indistinctly).


.. hint:: This container can be found in the :code:`third_party/managed_allocator.h` source.
