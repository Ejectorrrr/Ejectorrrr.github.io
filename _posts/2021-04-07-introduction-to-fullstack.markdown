---
layout: post
title: 全栈技术简介
category: 原创
---



简单梳理一下前后端技术。



## 目录

1. Web应用
2. 前端技术
3. 后端技术
4. 验证技术



## Web应用

前端负责展示，后端负责提供数据，它们合在一起就实现了一个Web应用，因此在介绍前后端技术之前，我们首先明确什么是Web应用。

一个Web应用的本质：

1. 浏览器发送一个HTTP请求
2. 服务器收到请求，生成一个HTML文档
3. 服务器把HTML文档作为HTTP响应的Body发送给浏览器
4. 浏览器收到HTTP响应，从HTTP Body取出HTML文档并显示

简单来说，围绕`HTML`相关的都是前端技术，围绕`HTTP`相关的都是后端技术。



## 前端技术

前端的运行环境是浏览器，浏览器执行HTML脚本，一个网页的展示过程包括获取HTML、解析HTML和渲染HTML，这里的HTML包含三部分内容：

- __HTML__ 定义了页面的 __内容__（和事件）
- __CSS__ 控制页面元素的 __样式__
- __JavaScript__ 负责页面的 __交互逻辑__

因此浏览器约等于一个 __HTML解析器__ + __JS引擎__ + __图形渲染器__。

浏览器渲染的HTML本身是由浏览器request后端服务器得到的，同时该HTML内部还可以嵌入更多的request，从而异步加载网页的不同位置内容。

#### JavaScript

JavaScript是处理用户与页面交互逻辑的事实标准，本质是一种运行在浏览器中的解释语言，具备跨平台、跨浏览器的特点。

运行JS脚本的主要方式是将JS代码或.js脚本地址嵌入到HTML中的`<script>`标签内。我的理解是，浏览器通过HTML中的`<script>`标签对外暴露了其内置的JS引擎。

页面如何通过JS实现与外部的交互？浏览器对外暴露了许多对象，通过读取和修改这些对象的属性值，就可以改变页面渲染的效果，这就是交互的原理。在JavaScript中，典型的浏览器对象包括window（浏览器窗口）、navigator（浏览器信息）、screen（屏幕）、location（当前页面的URL）、document（当前页面，即 __HTML解析得到的DOM根节点__）、history（浏览器历史记录）。

与页面状态关系最紧密的就是解析HTML得到的DOM对象，JavaScript中提供了一些操作DOM的利器，例如JQuery。

#### AJAX

由于JavaScript只能单线程运行（由引擎决定，而引擎遵循JS标准），面对HTML中的多个JS脚本，浏览器中的JS引擎也只能按顺序依次执行这些JS脚本。出于性能考虑，当JS脚本中涉及IO时都应该使用异步IO（无阻塞），AJAX就是对JavaScript中异步IO库的统称。

与前端交互的本质是修改浏览器对象，通过AJAX，就可以高效地完全依赖前端JS代码直接操作浏览器对象，而无需后端服务器反复重新生成整个HTML（多次加载页面）。

AJAX就是基于JS中的 __异步请求接口__ + __操作浏览器对象__ 的能力，实现页面内更新（而非重载整个页面）。



## 后端技术

从后端视角出发，整个Web应用有两种架构：

- __一体化架构__，服务器负责生成HTML
- __前后端分离（SOA）架构__，服务器负责对外暴露接口以提供数据

对于一体化架构，服务器生成的HTML既可以是在服务器上保存好的静态HTML，也可以是服务器动态生成的HTML。

无论一体化架构还是SOA架构，服务器的核心职责都是 __响应请求__，而HTTP协议的请求和响应遵循相同的格式，均包含`Header`和`Body`两部分，其中`Body`是可选的。

_GET请求_：

```http
GET /path HTTP/1.1
Header1: Value1
Header2: Value2
Header3: Value3
```

_POST请求_：

```http
POST /path HTTP/1.1
Header1: Value1
Header2: Value2
Header3: Value3

body data goes here...
```

_响应_：

```http
200 OK
Header1: Value1
Header2: Value2
Header3: Value3

body data goes here...
```

对于服务器接口，除了定义成REST，也可以是RPC。

#### 请求

`GET`仅请求资源，`POST`还会附带用户数据

一台服务器往往会响应多个不同的请求，不同请求间主要通过URL来区分调用服务器上的哪个函数，URL中的`domain`指定一台服务器的入口，URL中的`path`指定服务器根目录下的路径，用于区分不同的服务

#### 响应

为了响应不同请求，并灵活生成HTML，__服务端框架__ 应运而生，典型的有基于Python的Nginx、Flask、Django等，基于JavaScript的Node.js等。针对 __模版化地生成HTML__，以解耦Python代码和HTML代码，还衍生出`MVC`（Model-View_controller）这一设计模式。

`Node.js`不只是一个框架，还包含了完整的运行时和包管理工具`npm`。

当后端使用Node.js时，前后端代码都统一为JavaScript，实现了全栈技术的打通。



## 验证技术

#### Cookie

Cookie是由服务器生成并发送到客户端的key-value标示符。由于HTTP协议是无状态的，但是服务器要区分每个请求是由哪个用户发过来的，因此需要用Cookie来记录一些额外的状态。Cookie会过期，过期后会被删除。

#### Session

Session和Cookie的作用相似，都是为了保存额外的状态。__Cookie存储在浏览器端__，而 __Session存储在服务器端__，这么做的目的是保护敏感的隐私数据。

Session可以看作是对Cookie机制的一种前后端分离实现，即Cookie只保存session_id，而敏感的状态数据保存在服务端。

Session也有过期时间。

__Cookie/Session的问题__：

- Session最大的问题是会占用服务器资源

- Cookie和Session都无法处理前端页面和API不同源（URL中的domain不同）的问题

#### Token

服务端根据用户名/密码、时间戳等生成的加密字符串，用于区分已认证用户，保存在 __客户端__，有有效期，可以避开同源策略。

Token验证有一些标准实现，例如JWT等。



### 参考：

[廖雪峰的JavaScript教程][1]

[Token 认证的来龙去脉][2]

[1]:https://www.liaoxuefeng.com/wiki/1022910821149312
[2]:https://segmentfault.com/a/1190000013010835

