---
layout: post
title: 【翻译】用并发加速你的Python程序
category: 翻译
---



原文链接：https://realpython.com/python-concurrency/



目录：

- 什么是并发（concurrency）
- 什么是并行（parallelism）
- 什么时候该用并发
- 如何加速一个IO密集型程序
- 如何加速一个CPU密集型程序
- 什么时候使用并发
- 总结



在这篇文章中，你将了解下面的概念：

- 并发（concurrency）是什么
- 并行（parallelism）是什么
- Python中实现并发的方法对比（threading, asyncio, multiprocessing）
- 什么时候使用并发，使用哪个moodule



## 什么是并发（concurrency）

并发的字典释义是“同时发生”。在Python中，有很多名称（_thread_，_task_，_process_）对应“正在同时发生的事物”，不过从高层来看，它们都指一系列按顺序运行的指令。

我倾向于将这些同时进行的指令序列视为不同的“思维轨迹”（trains of thought）。每一条轨迹都可以在某个点暂停，原本处理该轨迹的大脑或CPU会切换到另一条轨迹上。每条轨迹中的状态（state）都被保留，确保它总可以从中断的地方重新开始。

你可能会好奇为什么Python使用了这么多不同的词汇来描述同一个概念。实际上，thread、task和process只是在high-level视角下相同。一旦你深入到细节，它们就会呈现出不同。通过本文的例子我们将会看到更多它们之间的区别。

现在我们开始讨论“同时发生”中的“同时”。你必须留意的地方是，只有multiprocessing在真正意义上同时运行多条“思维轨迹”，而threading和asyncio都运行在单核上，同一时间只有一条“思维轨迹”实际在运行。由于它们都能有效地轮流运行各条“思维轨迹”，从而加速整个程序执行过程，因此尽管它们不都能“同时”运行多条“思维轨迹”，我们仍然称之为并发（concurrency）。

threading和asyncio的主要区别就在于多任务轮换的方式。在threading中，操作系统掌控每一个线程（thread），可以在任意时间中断一个线程并运行另一个。这种由操作系统先发制人（preempt）地在线程间切换的方式成为__pre-emptive multitasking__。

__pre-emptive multitasking__的便利是，线程代码不需要针对切换做任何事。问题是，操作系统可决定在任意时间切换，甚至是在一条Python语句（statement）内切换，例如x = x + 1中。

asyncio则使用__cooperative multitasking__，任务自主决定何时可以被切换，这种方式在代码上需要一些额外工作，但优势是你能掌握任务在哪里进行切换。后面你会看到这一改变如何简化你的设计。

## 什么是并行（parallelism）

目前为止，我们了解的都是单核上的并发。如何利用计算机中的多核呢——multiprocessing。

利用multiprocessing，Python可以创建新的进程。每个进程都可以被视为一个完全不同的程序，通常从技术角度进程被定义为一个资源集合，包括内存、文件句柄等。一种理解进程的方式是，每个进程都运行在自己的Python解释器上。

在multiprocessing程序中，每个“思维轨迹”都运行在一个不同的核上，这意味着它们是真实意义上同时运行的。这样做也会带来一些复杂性，但是大部分时候Python都能够妥善处理。

现在你已经知道并发和并行是什么了，让我们来回顾下它们的区别，

| 并发类型                             | 切换策略                         | 处理器数量 |
| ------------------------------------ | -------------------------------- | ---------- |
| Pre-emptive multitasking (threading) | 操作系统决定何时切换             | 1          |
| Cooperative multitasking (asyncio)   | 任务决定何时移交控制权           | 1          |
| Multiproocessing (multiprocessing)   | 多个进程同时运行在不同的处理器上 | Many       |

接下来我们看不同的并发类型分别能加速什么类型的程序。

## 并发何时有效

并发对两类问题有显著影响：CPU密集型和IO密集型。

IO密集型会由于频繁等待外部资源导致程序变慢，外部资源的速度比CPU慢得越多，这种情况越严重。

比CPU慢的设备很多，不过大多不会与你的程序交互，其中交互最频繁的是文件系统和网络连接。



在上图中，蓝色框表示程序运行的时间，红色框表示等待IO完成的时间。网络请求比CPU指令运行的时间高出几个数量级，因此你程序的大部分时间都在等待。这就是浏览器在大部分时候所做的。

另一个反面是，CPU密集型程序需要做很多计算，而并不需要与文件或网络交互。限制程序速度的资源只有CPU。

这是CPU密集型程序的图。



下一节的例子会展示，不同形式的并发会使CPU密集型和IO密集型程序变好或变差。将并发添加到程序中会增加额外的代码和复杂性，你必须决定是否这样做是否能带来潜在的加速。在本文结束时，你将有足够的知识来作出决定。

这是一个快速总结：

| IO密集型进程                                               | CPU密集型进程                      |
| ---------------------------------------------------------- | ---------------------------------- |
| 程序花费太多时间与低速设备交互，例如网络连接、硬盘或打印机 | 程序大部分时候在执行CPU操作        |
| 加速的方式是将等待这些慢速设备的时间重叠起来               | 加速的方式是在同一时间执行更多计算 |

## 如何加速IO密集型程序

IO密集型程序的一个典型例子是：通过网络下载内容。

#### 同步版

我们首先来看该任务的非并发实现。示例程序需要`requests`模块，可以通过`pip install requests`安装，建议使用`virtualenv`的虚拟环境。下面的实现不包含任何并发：

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

正如你看到的，这是个很短的程序。`download_site()`从一个URL下载内容并打印其大小。要指出的一个小点是我们使用了`requests`中的`Session`对象。

也可以直接使用`requests`中的`get()`方法，但是创建一个`Session`对象会使`requests`做一些网络上的trick，带来速度上的提升。

`download_all_sites()`首先创建了`Session`对象，然后遍历`sites`列表，依次从每一个下载内容。最终，我们将过程的总耗时打印出来，这样便于我们比较引入并发后到底能带来多少提升。

这个程序的流程图基本就是上一节中IO密集型程序的示意图。

##### 为什么需要同步版本

这个代码的意义在于，它很简单。编写、调试和理解起来都很简单。由于只有一条思维轨迹，很容易预测

下一步会产生什么效果。

##### 同步版本的问题

主要的问题是与其他方式相比速度慢很多，下面是我们测试的结果：

```shell
$ ./io_non_concurrent.py
   [most output skipped]
Downloaded 160 in 14.289619207382202 seconds
```

慢不一定是个问题。如果一个程序的同步实现只耗时2秒且并不会被频繁执行，那就不值得为其增加并发实现。

#### `threading`版

尽管写线程化的程序需要更多的努力，但是对于简单的例子代价很小，下面就是一个例子：

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

添加`threading`不用改变程序的整体结构，只是`download_all_sites()`从对每个站点url调用一次函数变为一个更复杂的机构。

在这个版本中，首先创建了`ThreadPoolExecutor`，看起来有点复杂。我们来分解一下：`ThreadPoolExecutor = Thread + Pool + Executor`。

你已经了解了什么是`Tread`，即一条思维轨迹。`Pool`是真正有意思的地方，该对象将创建一池子线程，每一个都可以并发的执行。最终，`Executor`负责控制池子中的线程如何以及何时执行。在本例中，即执行池子中的请求。

标准库中已经实现了`TreadPoolExecutor`作为一个上下文管理器（context manager），因此你可以使用`with`语法来管理线程池的创建和释放。

基于`ThreadPoolExecutor`，你可以便捷地利用其中的`map()`方法。这个方法会对列表中的每个站点url执行传入的函数。最妙的地方是，它会基于所管理的一池子线程自动实现并发执行。

来自于其他语言或Python2的读者可能会好奇，那些使用`threading`时用于管理线程细节的常规对象和方法在哪里，例如`Thread.start()`，`Thread.join()`，`Queue`。

它们依然都在，你可以使用它们来实现细粒度的线程控制。但是从Python3.2开始，标准库添加了一个名为`Executors`的高层抽象，可以在你不需要如此细粒度控制时帮你管理许多细节。

我们例子中其他有趣的变化是，每个线程都需要创建自己的`Session`对象。这一点在你查看`requests`的文档时不容易注意到，但是读一下这个issue，你就会明白每个线程都需要一个独立的Session。

这是`threading`中一个有趣而又难以理解的issue，因为操作系统控制着你的任务何时中断以及另一个任务何时开始，任何在线程之间共享需要是线程安全的，不幸的是，`requests.Session()`不是线程安全的。

取决于数据及使用数据的方式，有几种策略可以确保数据访问是线程安全的。其中之一就是使用线程安全的数据结构，例如Python `queue`模块中的`Queue`。

这些对象利用低层设施，如`threading.Lock`，来确保同一时间只有一个线程可以访问某个代码块或内存位。通过`TreadPoolExecutor`对象，你实际上间接地使用了这个策略。

另一种策略称为“线程局部存储”（thread local storage）。`threading.local()`创建一个看起来像在全局但实际仅存在于单个线程内的对象。在本例中，通过`thread_local`和`get_session()`实现这一策略：

```python
thread_local = threading.local()


def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session
```

在实现中，你只需创建一次`local()`对象，而不是在每个线程中创建一次，该对象会负责将来自不同线程的访问发送到不同的数据上。

当`get_session()`被调用，会返回针对当前运行线程的session对象。因此每个线程在首次调用`get_session()`后都会创建一个独立的session，之后就会一直使用这个session。

最后，设置线程数的注意事项。在上面的例子中我们使用了5个线程。请大胆尝试调整这个数字，并观察整体运行时间有什么变化。或许你期望每个下载都对应一个线程是最快的，但至少在我们的系统中，实际情况不是这样的。我发现最快的方式是设置5到10个线程。如果把线程数进一步调大，创建和销毁线程的额外花销会超过任何节省出来的时间。

正确的线程数并不是一成不变的，需要通过一些实验来确定。

##### 为什么`threading`版本有效

这个实现很快！这是我们测试的最快结果，回一下，非并发版本花费了14秒：

```shell
$ ./io_threading.py
   [most output skipped]
Downloaded 160 in 3.7238826751708984 seconds
```

下图是对应的流程图：



它在同一时刻使用多个线程开启多个网络访问，因此将等待时间重叠，获得了更快的速度。

##### `threading`版本的问题

正如你在例子中看到的，代码变多了一点，你也需要思考哪些数据需要在线程间共享。

线程可以以不易察觉的方式进行交互。这些交互会造成竞态条件，引发随机的、难以查找的bug。不熟悉竞态条件改变的读者可以进行扩展阅读。

Race Conditions

#### `asyncio`版实现

在你开始阅读`asyncio`的实例代码前，我们先看看`asyncio`如何工作。

##### `asyncio`基础

本节我们简要描述`asyncio`，重点是说明它如何工作。

`asyncio`是，一个称为事件循环的Python对象，控制每个任务如何以及何时执行。

