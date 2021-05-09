---
layout: post
title: 【翻译】用并发加速你的Python程序
category: 翻译
---

本篇是译文，原文链接：https://realpython.com/python-concurrency/

---

[toc]

在这篇文章中，你将了解下面的概念：

- __并发__（__concurrency__）是什么
- __并行__（__parallelism__）是什么
- Python中实现并发的方法对比（`threading`, `asyncio`, `multiprocessing`）
- 何时使用 __并发__ 以及应使用哪个module

## 1. 什么是并发（concurrency）

__并发__ 的字典释义是“同时发生”。在Python中，有很多名词（_thread_，_task_，_process_）意指“正在同时发生的事物”，不过从高级（high-level）的角度看，它们都指`一系列按顺序运行的指令`。

我倾向于将这些同时进行的“__指令序列__”视为不同的“__思路__”（trains of thought）。每一条“思路”都可以在某个点暂停，原本处理该“思路”的大脑或CPU会切换到另一条“思路”上。每条“思路”中的状态（state）都会被保留，确保它总可以从中断的地方重新开始。

你可能会好奇为什么Python使用了这么多不同的词汇来描述同一个概念。实际上，`thread`、`task`和`process`只是在高级（high-level）视角下相同，一旦你深入到细节就会发现它们的不同。我们将通过本文的例子看到更多它们之间的区别。

现在我们讨论“__同时发生__”的概念。须留意的是，只有`multiprocessing`是真正意义上 __同时运行多条思路__，而`threading`和`asyncio`都运行在单核上，__同一时间只实际运行一条思路__，通过在各条“思路”间轮流切换来加速整个程序的执行。尽管它们不都能同时运行多条“思路”，我们仍然称之为 __并发__（__concurrency__）。

`threading`和`asyncio`的主要区别就在于其多任务轮换的方式。在`threading`中，操作系统掌控每一个线程（thread），可以在任意时间中断一个线程并运行另一个。这种由操作系统 __先发制人__（__preempt__）地在线程间切换的方式称为 __pre-emptive multitasking__。

__pre-emptive multitasking__ 的便利是，线程代码不需要针对切换做任何事。问题是，操作系统可决定在任意时间切换，甚至是在一条Python语句（statement）内切换，例如x = x + 1中。

`asyncio`则使用 __cooperative multitasking__，任务自主决定何时可以被切换，这种方式在代码上需要一些额外工作，但优势是你能掌握任务在哪里进行切换。后面你会看到这一改变如何简化你的设计。

## 2. 什么是并行（parallelism）

目前为止，我们了解的都是单核上的并发（`threading`和`asyncio`）。如何利用计算机中的多核呢——`multiprocessing`。

利用`multiprocessing`，Python可以创建新的进程。每个进程都可以被视为一个完全独立的程序，从技术角度，进程是一个包括内存、文件句柄等的资源集合。一种理解进程的方式是，每个进程都运行在自己单独的Python解释器上。

在`multiprocessing`程序中，每条“思路”都可以运行在一个不同的核上，这意味着它们是真实意义上 __同时运行__ 的。这样做也会带来一些复杂性，但是大部分时候Python都能够妥善处理。

现在你已经了解 __并发__ 和 __并行__ 是什么了，让我们来回顾下它们的区别，

| 并发类型                               | 切换策略                         | 处理器数量 |
| -------------------------------------- | -------------------------------- | ---------- |
| Pre-emptive multitasking (`threading`) | 操作系统决定何时切换             | 1          |
| Cooperative multitasking (`asyncio`)   | 任务决定何时移交控制权           | 1          |
| Multiprocessing (`multiprocessing`)    | 多个进程同时运行在不同的处理器上 | Many       |

接下来我们看不同的并发类型分别能加速什么类型的程序。

| 编者注：并发（concurrency）是逻辑并行，并行（parallelism）是物理并行。

## 3. 并发何时有效

并发对两类问题有显著影响：__CPU密集型__ 和 __IO密集型__。

密集的IO会使程序频繁等待外部资源而变慢，外部资源的速度比CPU慢得越多，这种情况越严重。

比CPU慢的设备很多，不过大多不会与你的程序交互，其中交互最频繁的是文件系统和网络连接。

![](/Users/wangchenyang3/SourceRepos/Own/Ejectorrrr.github.io/_posts/images/speed-up-your-python-program-with-concurrency/3-1.jpg)

在上图中，蓝色框表示程序运行的时间，红色框表示等待IO完成的时间。网络请求比CPU指令运行的时间高出几个数量级，因此你的程序大部分时间都在等待。这就是浏览器在大部分时间所做的。

另一个反面是，CPU密集型程序需要做很多计算，而并不需要与文件或网络交互，限制程序执行速度的资源就只有CPU。以下是CPU密集型程序的图。

![](/Users/wangchenyang3/SourceRepos/Own/Ejectorrrr.github.io/_posts/images/speed-up-your-python-program-with-concurrency/3-2.jpg)

下一节的例子会展示，不同形式的并发会使CPU密集型和IO密集型程序变好或变差。将并发添加到程序中会增加额外的代码和复杂性，你必须决定这样做是否能带来潜在的加速。在本文结束时，你将有足够的知识来作出决定。

这是一个快速总结：

| IO密集型进程                                               | CPU密集型进程                      |
| ---------------------------------------------------------- | ---------------------------------- |
| 程序花费太多时间与低速设备交互，例如网络连接、硬盘或打印机 | 程序大部分时候在执行CPU操作        |
| 加速的方式是将等待这些慢速设备的时间重叠起来               | 加速的方式是在同一时间执行更多计算 |

## 4. 如何加速IO密集型程序

IO密集型程序的一个典型例子是：通过网络下载内容。

#### 同步版

我们首先来看该任务的非并发实现。示例程序需要`requests`模块，可以通过`pip install requests`安装，建议使用`virtualenv`虚拟环境。下面的实现不包含任何并发：

```python
import requests
import time

def download_site(url, session):
    with session.get(url) as response:
        print(f"Read {len(response.content)} from {url}")

def download_all_sites(sites):
    with requests.Session() as session:
        for url in sites:
            download_site(url, session)

if __name__ == "__main__":
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} in {duration} seconds")
```

正如你看到的，这个程度很短。`download_site()`从一个URL下载内容并打印其大小。要指出的一个小点是我们使用了`requests`中的`Session`对象，也可以直接使用`requests`中的`get()`方法，但是创建一个`Session`对象会使`requests`做一些网络上的trick，带来速度上的提升。

`download_all_sites()`首先创建了`Session`对象，然后遍历`sites`列表，依次从每一个下载内容。最终，我们将过程的总耗时打印出来，这样便于我们比较引入并发后到底能带来多少提升。

这个程序的流程图基本就是上一节中IO密集型程序的流程图。

##### 为什么同步版有效

这个代码的意义在于，它很简单。编写、调试和理解起来都很简单。由于只有一条思维轨迹，很容易预测

下一步会产生什么效果。

##### 同步版的问题

主要的问题是与其他方式相比速度慢很多，下面是我们测试的结果：

```shell
$ ./io_non_concurrent.py
   [most output skipped]
Downloaded 160 in 14.289619207382202 seconds
```

慢不一定是个问题。如果一个程序的同步实现只耗时2秒且不会被频繁执行，那就不值得为其增加并发实现。

#### `threading`版

下面是一个线程化（threaded）实现的例子：

```python
import concurrent.futures
import requests
import threading
import time

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

def download_site(url):
    session = get_session()
    with session.get(url) as response:
        print(f"Read {len(response.content)} from {url}")

def download_all_sites(sites):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_site, sites)

if __name__ == "__main__":
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} in {duration} seconds")
```

添加`threading`不用改变程序的整体结构，只是将`download_all_sites()`从对每个url调用一次函数变为一个更复杂的结构。

在这个版本中，首先创建了`ThreadPoolExecutor`，看起来有点复杂。我们来分解一下：`ThreadPoolExecutor = Thread + Pool + Executor`。

`Tread`就是我们之前提到的“思路”。`Pool`是真正有意思的地方，它将创建一池子线程，每一个都可以并发执行。`Executor`负责控制池子中的线程如何以及何时执行。

| 编者注：`Tread`承载业务逻辑，`Pool`分配线程资源，`Executor`负责调度。

标准库中已经将`TreadPoolExecutor`实现为上下文管理器（context manager），因此你可以使用`with`语法来管理线程池的创建和释放。

`ThreadPoolExecutor`提供了`map()`方法，该方法会对列表中的每个url执行传入的函数。最妙的地方是，它会基于线程池自动实现并发执行。

来自于其他语言或Python2的读者可能会好奇，那些使用`threading`时用于管理线程细节的常规对象和方法在哪里，例如`Thread.start()`，`Thread.join()`，`Queue`。它们依然都在，你可以使用它们来实现细粒度的线程控制。但是从Python3.2开始，标准库添加了一个名为`Executors`的高层抽象，可以在你不需要如此细粒度控制时帮你管理许多细节。

我们例子中其他有趣的变化是，每个线程都需要创建自己的`Session`对象。这一点在你查看`requests`的文档时不容易注意到，但是读一下这个[issue][https://github.com/requests/requests/issues/2766]，你就会明白每个线程都需要一个独立的Session。这是`threading`中一个有趣而又难以理解的issue，因为操作系统控制着你的任务何时中断以及另一个任务何时开始，所以任何在线程之间共享的数据必须是线程安全的。不幸的是，__`requests.Session()`不是线程安全的__。

取决于数据及使用数据的方式，有几种策略可以确保数据访问是线程安全的。其中之一就是使用线程安全的数据结构，例如Python `queue`模块中的`Queue`。

这些对象利用低层设施，如`threading.Lock`，来确保同一时间只有一个线程可以访问某个代码块或内存位。通过`TreadPoolExecutor`对象，你实际上间接地使用了这个策略。

另一种策略称为“__线程局部存储__”（thread-local storage）。`threading.local()`创建一个看起来像在全局但实际仅存在于单个线程内的对象。在本例中，通过`thread_local`和`get_session()`实现这一策略：

```python
thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session
```

在实现中，你只需创建一次`local()`对象，而不是在每个线程中创建一次，该对象会负责将来自不同线程的访问发送到不同的数据上。

当`get_session()`被调用，会返回针对当前运行线程的session对象。因此 __每个线程在首次调用`get_session()`后都会创建一个独立的session__，之后就会一直使用这个session。

最后，设置线程数的注意事项。在上面的例子中我们使用了5个线程。请大胆尝试调整这个数字，并观察整体运行时间有什么变化。或许你期望每个下载都对应一个线程是最快的，但至少在我们的系统中，实际情况不是这样的。我发现最快的方式是设置5到10个线程。如果把线程数进一步调大，创建和销毁线程的额外花销会超过任何节省出来的时间。

正确的线程数并不是一成不变的，需要通过一些实验来确定。

##### 为什么`threading`版有效

这个实现很快！这是我们测试的最快结果，回忆一下非并发版本花费了14秒：

```shell
$ ./io_threading.py
   [most output skipped]
Downloaded 160 in 3.7238826751708984 seconds
```

下图是对应的时序流程图：

![](/Users/wangchenyang3/SourceRepos/Own/Ejectorrrr.github.io/_posts/images/speed-up-your-python-program-with-concurrency/4-1.jpg)

它在同一时刻使用多个线程开启多个网络访问，因此将等待时间重叠，获得了更快的速度。

##### `threading`版的问题

正如你在例子中看到的，代码变多了一点，还需要思考哪些数据需要在线程间共享。

线程间可能以不易察觉的方式进行交互，这些交互会造成竞态条件（race conditions），引发随机的、间歇的、难以查找的bug。不熟悉竞态条件的读者可以阅读以下扩展内容。

Race Conditions

> 竞态条件是一类在多线程代码中经常发生的隐秘bug，一般由于开发者没能充分保护数据访问，从而招致线程间互相干扰。在写多线程代码时需要额外的步骤来确保线程安全。
>
> 操作系统控制着线程的执行和切换，线程切换可能发生在任何时间点，甚至是一条Python语句的内部（sub-steps of a Python statement），下面是一个例子：
>
> ```python
> import concurrent.futures
> 
> counter = 0
> 
> def increment_counter(fake_value):
>     global counter
>     for _ in range(100):
>         counter += 1
> 
> if __name__ == "__main__":
>     fake_data = [x for x in range(5000)]
>     counter = 0
>     with concurrent.futures.ThreadPoolExecutor(max_workers=5000) as executor:
>         executor.map(increment_counter, fake_data)
> ```
>
> 这个代码的结构与前述`threading`的例子类似，区别在于每个线程都访问同一个全局变量`counter`，而`counter`没有受到任何保护，因此不是线程安全的。
>
> 为了累加`counter`，每个线程都要读取它的当前值、加1、再将结果存回原变量，这些都发生在`counter += 1`这条语句中。
>
> 由于操作系统对你的代码一无所知并且可能在任意时间点切换线程，因此切换很有可能发生在当前线程读取了`counter`值之后以及将更新值写回`counter`之前。如果切换后另一线程中的代码也修改了`counter`的值，那么原线程就维护了一个过期值，那么问题就产生了。
>
> 正如你想象的，恰好出现这一情况还是很罕见的，可能运行了上千次程序也不会遇到这个问题。这也使得这类问题很难被debug，因为它很难重现。更进一步的例子，你需要知道`requests.Session()`不是线程安全的，这意味着当多个线程使用同一个`Session`时上面描述的情况就可能出现。



#### `asyncio`版

在你开始阅读`asyncio`的示例代码前，我们先看看`asyncio`如何工作。

##### `asyncio`基础

本节我们简要描述`asyncio`，重点说明它如何工作。

`asyncio`是一个被称为 __事件循环__（event loop）的Python对象，控制每个任务如何以及何时执行。事件循环掌握着每个任务及其当前状态。

一个任务可能有多种状态，为了简化，我们考虑一个只有2个状态的事件循环。"ready"状态表明一个任务已经准备好被执行，"waiting"状态表明任务正在等待外部事件完成，例如网络操作。

该简化事件循环为每种状态维护了一个任务列表。当事件循环选择一个"ready"任务开始执行后，该任务将拥有完全的控制权直到它主动将控制权交还给事件循环。

当运行中的任务将控制权交还给事件循环时，事件循环将该任务放入"ready"或"waiting"列表，然后检查"waiting"列表中的每个任务是否已经进入"ready"状态。

一旦所有的任务被再次放入正确的列表中，事件循环会执行下一个任务，整个过程如此反复。对于简化事件循环，它会选择等待时间最长的"ready"任务来执行，不断循环该过程直到整个事件循环完成。

`asyncio`的一个重要特性是 __除非主动放弃，否则任务不会失去控制权__。它们不会在一个操作的中途被中断，这使得在`asyncio`中共享资源比在`threading`中更容易，不需要担心代码是否线程安全。

以上是对`asyncio`原理的顶层介绍，更多细节可以参考这个[回答](https://stackoverflow.com/a/51116910/6843734)。

##### async 和 await

现在我们讨论Python中新加入的两个关键字：`async`和`await`。根据上面的讨论，你可以将`await`视为一种能使任务将控制权交还给事件循环的魔法。当你的代码`await`一个函数调用时，意味着该调用应负责释放控制权。

可以将`async`简单理解为一个传递给Python解释器的标记，告知其下面定义的函数将使用`await`。虽然也有一些情况并不如此，例如[异步generator](https://www.python.org/dev/peps/pep-0525/)，但这仍是一个适用于大部分情况的简单范式。

一种意外是`async with`语句，它会针对你要`await`的对象创建一个上下文管理器（context manager）。尽管语义有所不同，但思想是一致的：将该上下文管理器标记为可以被切换的。

管理事件循环和任务的交互看起来有一定的复杂性，不过对于刚开始接触`asyncio`的开发者而言，这些细节不重要，你需要记住 __任何调用`await`的函数都应该被`async`标记__，否则会收到语法错误。

##### 回到代码

现在你已经对`asyncio`有了基本了解，我们来看看`asyncio`版的示例代码。注意该版本添加了`aiohttp`，在运行代码前请首先运行`pip install aiohttp`：

```python
import asyncio
import time
import aiohttp

async def download_site(session, url):
    # 各线程共用一个session（未使用线程局部存储）
    async with session.get(url) as response:
        print("Read {0} from {1}".format(response.content_length, url))

async def download_all_sites(sites):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in sites:
            task = asyncio.ensure_future(download_site(session, url))
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.time()
    asyncio.get_event_loop().run_until_complete(download_all_sites(sites))
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} sites in {duration} seconds")
```

这一版本比之前的两个版本都要复杂，让我们从顶部开始。

###### download_site()

`download_site()`与`threading`版本中的几乎相同，除了修饰函数定义的`async`关键字和`session.get()`前的`async with`。稍后你将了解到为什么`session`可以不需线程局部存储而直接使用。

###### download_all_sites()

`download_all_sites()`与`threading`版本的区别最大。

本例将session创建为一个上下文管理器，所有任务共享该session。由于 __所有任务都运行在同一个线程中__，因此它们可以共享一个session。当session处于坏状态时，任何一个任务都没有办法中断另一个。

在上下文管理器内，由`asyncio.ensure_future()`创建了一个任务列表（编者注：每个任务就是一个`download_site()`），然后将这些任务启动。一旦所有任务都创建了，该函数使用`asyncio.gather()`来保持session在所有任务都完成前都是活跃的。

`threading`版代码也做了类似的工作，不过细节都被`ThreadPoolExecutor`一手包办了。目前还没有一个`AsyncioPoolExecutor`类。

还有一个小但是重要的改变，记得我们讨论过要创建多少个线程吗？在`threading`版代码中最优线程数并不明确，而`asyncio`比`threading`的扩展性更强。创建任务比创建线程所占用的资源和时间更少，因此可以创建和运行更多任务。本例中为每个站点都创建了一个独立的任务，工作地十分良好。

###### \_\_main\_\_

运用`asyncio`的本质就是开始一个事件循环并告诉它运行哪些任务。`get_event_loop()`和`run_until_complete`两个函数的名称已经表明了它们的意图。如果你已经更新了Python3.7，Python核心开发者简化了这个语法，不必使用`asyncio.get_event_loop().run_until_complete()`，直接使用`asyncio.run()`。

##### 为什么`asyncio`版有效

它确实很快！在我们机器的测试中，这是最快的版本：

```shell
$ ./io_asyncio.py
   [most output skipped]
Downloaded 160 in 2.5727896690368652 seconds
```

执行时间的流程图看起来和`threading`版代码的很像，区别在于 __I/O请求都是由同一个线程发出的__：

![](/Users/wangchenyang3/SourceRepos/Own/Ejectorrrr.github.io/_posts/images/speed-up-your-python-program-with-concurrency/4-2.jpg)

缺少类似`ThreadPoolExecutor`的包装类（wrapper）使得代码比`threading`版更复杂。这倒是恰好说明，为了获得更好的性能往往需要额外的工作。额外复杂度的另一收益是，它强迫你去思考一个给定的任务何时会被切换，这会帮助你创造一个更好更快的设计。

扩展性的问题在这里也被突出。在`threading`版中，为每个站点创建一个线程会比只使用几个线程显著变慢，而`asyncio`版运行上百个任务也完全不会变慢。

##### `asyncio`版的问题

`asyncio`也有一些问题。__`asyncio`需要配合一些库的`async`版本才能发挥其全部优势__。倘若你使用`requests`来下载站点就会很慢，因为`requests`无法通知事件循环它被阻塞了（编者注：`requests`不是一个`async`实现）。这一问题随着时间推移将会越来越小，越来越多的库会拥抱`asyncio`。

另一个更微妙的问题是，__任何一个任务出现问题都会使 _cooperative multitasking_ 的全部优点消失殆尽__。如果代码中的一个小错误导致一个任务跑飞，长时间占用处理器无法退出，那么其他任务就都无法运行。对于事件循环，只要任务没有交换控制权，它就没有办法插手。

考虑到这些问题，让我们来看另一种完全不同的并发实现方法，`multiprocessing`。

#### `multiprocessing`版

不同于之前的方法，`multiprocessing`版能够完全利用计算机中的多个CPU，我们先看代码：

```python
import requests
import multiprocessing
import time

session = None

def set_global_session():
    global session
    if not session:
        session = requests.Session()

def download_site(url):
    with session.get(url) as response:
        name = multiprocessing.current_process().name
        print(f"{name}:Read {len(response.content)} from {url}")

def download_all_sites(sites):
    with multiprocessing.Pool(initializer=set_global_session) as pool:
        pool.map(download_site, sites)

if __name__ == "__main__":
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} in {duration} seconds")
```

这个代码比`asyncio`版更短，看起来和`threading`版很像。在我们分析代码前，先快速浏览下`multiprocessing`能够为你做什么。

##### `multiprocessing`速览

截至目前，所有的并发示例都运行在单核上。这一切的原因都要归结为CPython的设计，一种称为GIL（Global Interpreter Lock）的机制。本文不会深入[GIL](https://realpython.com/python-gil/)的细节，我们只需要知道同步版、`threading`版和`asyncio`版三种实现都是运行在单核上的就够了。

标准库中的`multiprocessing`就是为了打破这一障碍，使代码能够运行在多个CPU上。从高层上理解，它在每个CPU上都创建了一个新的Python解释器，然后将代码的各部分分散到各解释器中运行。

正如你猜想的，创建一个单独的解释器远没有在现有的解释器中创新新线程来得快。尽管这是一个很重的操作，不过对于合适的问题，它会带来显著的变化。

##### 回到代码

主要变化是`download_all_sites()`，相比同步版，没有反复调用`download_site()`，而是创建了一个`multiprocessing.Pool`对象，将`download_site()`映射到代表站点的iterable对象。在`threading`版中也有类似的实现。

`Pool`会创建多个独立的Python解释器进程，每个进程会针对iterable对象（即站点列表）中的部分元素运行指定的函数。主进程和其他进程间的通讯由`multiprocessing`模块负责。

注意在创建`Pool`的这一行代码中并没有指定创建多少个进程，尽管这是一个可选参数。`multiprocessing.Pool`会默认选择计算机中的CPU数。实际上，增加进程数并不会使程序更快，反而会使程序变慢，因为建立和销毁这些进程的开销比并行I/O获得的收益更大。

在创建`Pool`时还设置了`initializer=set_global_session`，`Pool`中的每个进程都有自己的内存空间，这意味着它们无法共享一个`Session`对象。我们希望在每个进程中只创建一次`Session`，而不是每次调用函数的时候都重新创建，这就是`initializer`参数的用途。

代码剩下的部分都和之前的基本类似。

##### 为什么`multiprocessing`版有效

`multiprocess`版的代码只需要很少的额外代码，并且能够充分利用计算机中的CPU资源。执行时间的流程图如下：

![](/Users/wangchenyang3/SourceRepos/Own/Ejectorrrr.github.io/_posts/images/speed-up-your-python-program-with-concurrency/4-3.jpg)

##### `multiprocessing`版的问题

你需要花一些时间思考在每个进程中哪些变量会被访问到。

另外，运行速度比`asyncio`和`threading`版要慢：

```shell
$ ./io_mp.py
    [most output skipped]
Downloaded 160 in 5.718175172805786 seconds
```

这并不惊奇，因为`multiprocess`并不主要针对I/O密集型的问题，在下一节中，我们将看到`multiprocessing`在CPU密集型问题中的更多应用。

## 5. 如何加速CPU密集型程序

截至目前的例子都是处理I/O密集型的问题，正如你看到的，一个I/O密集型问题将大部分时间花费在等待外部操作上，例如网络传输。而对一个CPU密集型的问题，几乎没有I/O操作，其总执行时间取决于处理所需数据的速度有多快。

作为示例，我们创建了一个能够长时间占用CPU的函数：

```python
def cpu_bound(number):
    return sum(i * i for i in range(number))
```

当你传入一个较大的数字时，该函数就会执行一段时间。

#### 同步版

我们首先来看非并发版：

```python
import time

def cpu_bound(number):
    return sum(i * i for i in range(number))

def find_sums(numbers):
    for number in numbers:
        cpu_bound(number)

if __name__ == "__main__":
    numbers = [5_000_000 + x for x in range(20)]

    start_time = time.time()
    find_sums(numbers)
    duration = time.time() - start_time
    print(f"Duration {duration} seconds")
```

该代码调用了20次`cpu_bound()`，每次都传入一个不同的大数。整个过程都运行在单核单进程单线程上，执行时间的流程图如下：

![](/Users/wangchenyang3/SourceRepos/Own/Ejectorrrr.github.io/_posts/images/speed-up-your-python-program-with-concurrency/5-1.jpg)

不同于I/O密集型，CPU密集型的运行时间都基本一致：

```shell
$ ./cpu_non_concurrent.py
Duration 7.834432125091553 seconds
```

我们来看看如何做得更好。

#### `threading`和`asyncio`版

你认为用`threading`和`asyncio`来重写这份代码能加速多少？

如果你回答“完全不会加速”，请奖励自己一块饼干。如果你回答“会变慢”，请奖励自己两块饼干。

在I/O密集型问题中，整体时间大部分花费在了等待慢操作上。`threading`和`asyncio`通过重叠这些等待时间而不是串行等待来实现加速。

在CPU密集型问题中，不存在等待，CPU总是会尽快完成任务。在Python中，多个线程和任务都在同一个进程中使用同一个CPU运行，这意味着一个CPU既要执行代码，还要承担建立线程或任务的额外工作，因此总耗时超过了10s：

```shell
$ ./cpu_threading.py
Duration 10.407078266143799 seconds
```

`threading`版的实现放在了[GitHub repo](https://github.com/realpython/materials/tree/master/concurrency-overview)中，可以自行测试。

#### `multiprocessing`版

终于到了见证`multiprocessing`闪耀的时候了。不同于其他并发库，`multiprocessing`被明确地设计用于在多CPU间分摊负载，下面是执行时间的流程图：

![](/Users/wangchenyang3/SourceRepos/Own/Ejectorrrr.github.io/_posts/images/speed-up-your-python-program-with-concurrency/5-2.jpg)

下面是代码：

```python
import multiprocessing
import time

def cpu_bound(number):
    return sum(i * i for i in range(number))

def find_sums(numbers):
    with multiprocessing.Pool() as pool:
        pool.map(cpu_bound, numbers)

if __name__ == "__main__":
    numbers = [5_000_000 + x for x in range(20)]

    start_time = time.time()
    find_sums(numbers)
    duration = time.time() - start_time
    print(f"Duration {duration} seconds")
```

相比非并发版中循环调用`cpu_bound()`，该代码中创建了`multiprocessing.Pool`对象，并使用`map()`方法将每个数发送给一个空闲进程。这和我们在I/O密集型问题中使用`multiprocessing`的方式一致。

你可以定义想要在`Pool`中创建和管理多少个`Process`对象，默认地，这将由你计算机中的CPU数决定，一个进程对应一个CPU。

另外，回忆一下我们在关于`threading`的第一部分提到的，`multiprocessing.Pool`是基于`Queue`和`Semaphore`建立的，接触过其他语言中多线程和多进程代码的同学对此会比较熟悉。

##### 为什么`multiprocessing`版有效

`multiprocess`版只需要很少的额外代码，相对容易建立，能够完全利用计算机中的CPU算力。

上次我们总结`multiprocessing`时其实已经说过这几点了，不过在本例中我们收获了极大的性能提升：

```shell
$ ./cpu_mp.py
Duration 2.5175397396087646 seconds
```

##### `multiprocessing`版的问题

使用`multiprocessing`依然存在缺点。在这个简单的例子中不明显，不过在实际问题中，要将问题分解为能在每个处理器上独立运行，有时是很难的。

另外，许多解决方案还要求进程间通信，这也会增加复杂度。

## 6. 何时使用并发

我们来总结一下关键概念，然后讨论一些决策要点，以帮助我们决定在项目中应使用哪一个并发模块（concurrency module）。

第一步，判断是否应该使用并发模块。尽管本文中的例子让这些库看起来都很简单，但并发通常伴随着额外的复杂度，并且可能导致难以发现的bug。除非有明确的性能问题，不要轻易增加并发。正如[Donald Knuth](https://en.wikipedia.org/wiki/Donald_Knuth)说的，“过早的优化是编程中所有魔鬼的根源”

第二步，一旦决定要优化程序，搞清楚你的程序是CPU密集型还是I/O密集型。记住，I/O密集型程序花费大部分时间等待CPU计算之外的事，而CPU密集型程序花费大量时间处理数据。

正如你所见，CPU密集型问题只能使用`multiprocessing`来加速，`threading`和`asyncio`都完全帮不上忙。

对于I/O密集型问题，在Python社区中有一条通用原则：“__Use `asyncio` when you can, `threading` when you must__”。`asyncio`可以提供最大的提速，但有时你需要的一些关键库可能没有 _async_ 实现。记住，任一尚未交还控制权的任务将阻塞所有其他任务。

## 7. 结论

现在你已经了解了Python中并发的基本类型：

- `threading`
- `asyncio`
- `multiprocessing`

你也了解了对特定问题应使用哪一种并发方法，以及使用并发可能出现的问题。

希望你从这篇文章中学到很多，并在你的项目中找到并发的用武之地。务必参加下面的“Python并发”测验来检查你的学习：

[Take the Quiz](https://realpython.com/quizzes/python-concurrency/)

