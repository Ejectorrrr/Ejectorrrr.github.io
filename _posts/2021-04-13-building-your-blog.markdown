---
layout: post
title: 创建个人博客
category: 原创
---



本篇记录一下博客搭建的流程。



## Tools

- Writer：typora

- Hosting service：GitHub Pages

- Site generator：Jekyll



## Introduction

GitHub Pages本质上是GitHub repository + site generator，利用它既可以为你的Github账号创建个人主页，也可以为账号下的每个project创建项目主页。

默认情况下，个人主页的地址都是`<username>.github.io`，项目主页的地址都是`<username>.github.io/<repository>`。

通过GitHub Pages发布站点时，都要指定站点文件所在的源，包括`仓库名`、`分支名`和`目录名`三个部分，且`目录名`只能是分支下的`/`或`/docs`目录。对于个人站点而言，默认的源是名为`<username>.github.io`的仓库下的`master`分支；对于项目站点而言，默认的源是项目所在仓库的`gh-pages`分支。

站点是一个静态网页，本质上只能通过HTML脚本（包含嵌入其中的CSS和JS脚本）发布，但我们的博客是利用markdown等标记语言编写的，因此在 __撰写博客__ 和 __发布页面__ 之间还需要一个转换器，这就是site generator的功能。GitHub Pages默认使用 __Jekyll__ 作为site generator。如果我们要使用另外的generator，需要在站点挂靠的发布源目录下添加`.nojekyll`文件，然后在本地使用generator build整个源目录，再将build之后的工程push到发布源上。作为对比，直接使用Jekyll可以省略本地build这一步。

完整的介绍在[这里][1]。



## Procedures

#### Prerequisite：

- Ruby
- Jekyll
- Bundler

Jekyll是一个`Ruby Gem`，`Gem`就是Ruby中的package，同时它也指Ruby的包管理软件（`RubyGems`），通过`gem install package-name`可以安装Ruby package。

`Gemfile`记录了一个工程所依赖的全部`Gem`。

Bundler是另一个`Ruby Gem`，专门负责批量安装`Gemfile`中的所有`Gem`。



#### 以创建个人站点为例：

1. [创建与域名`<username>.github.io`同名的仓库][3]
2. [配置页面的发布源][2]
3. 在本地clone上面创建的仓库
4. 进入本地与发布源一致的目录后，利用Jekyll创建初始化站点工程，即运行`jekyll new .`
5. 在`_posts/`目录下添加博客文章
6. 本地测试：先运行`bundle install`安装工程的依赖包，再运行`bundle exec jekyll serve`来build整个工程，最后打开[http://localhost:4000](http://localhost:4000/)检查效果
7. 提交本地的所有变更，并将本地提交记录push到远端仓库
8. 登录`<username>.github.io`检查上线效果

[可参照官方流程][4]



#### 注意

Jekyll不会build下列文件或文件夹：

- `/node_modules`或`/vendor`
- 以`_`，`.`，`#`开头的
- 以`~`结尾的
- 在`_config.yml`的`exclude`配置项下的



[1]:https://docs.github.com/en/pages/getting-started-with-github-pages
[2]:https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#choosing-a-publishing-source
[3]:https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site#creating-a-repository-for-your-site
[4]:https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll