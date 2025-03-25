抓取电商商品数据并规范化存储

pythonCopy Code
import requests
from bs4 import BeautifulSoup
import pandas as pd

# 多页数据抓取（带异常重试）
def scrape_products(base_url, max_pages=5):
    products = []
    for page in range(1, max_pages+1):
        try:
            response = requests.get(f"{base_url}?page={page}", timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')
            
            # 提取商品卡片
            cards = soup.select('div.product-card')
            for card in cards:
                # 清洗价格数据（处理货币符号）
                price = card.find('span', class_='price').text
                price = float(price.strip('¥$').replace(',', ''))
                
                # 构建结构化数据
                products.append({
                    'title': card.h3.text.strip(),
                    'price': price,
                    'rating': float(card['data-rating']),
                    'category': card['data-category']
                })
        except (requests.Timeout, AttributeError) as e:
            print(f"第{page}页抓取失败: {str(e)}")
    
    # 转换为DataFrame并去重
    return pd.DataFrame(products).drop_duplicates(subset=['title'])

# 执行抓取
df = scrape_products("https://example.com/products")



Pandas数据分析（进阶版）

‌场景‌：分析服务器性能监控数据

pythonCopy Code
import pandas as pd

# 读取监控数据
df = pd.read_csv('server_metrics.csv', parse_dates=['timestamp'])

# 1. 处理缺失值
df.fillna({'cpu_usage': df['cpu_usage'].median()}, inplace=True)

# 2. 按时间聚合
hourly_stats = df.resample('1H', on='timestamp').agg({
    'cpu_usage': 'mean',
    'memory_usage': 'max'
})

# 3. 筛选异常时段
high_load = df[(df['cpu_usage'] > 90) & (df['memory_usage'] > 85)]

# 4. 输出结果
print("每小时统计:\n", hourly_stats)
print("\n高负载时段:\n", high_load[['timestamp', 'host']])



CSV文件处理（基础版）

‌场景‌：处理服务器访问日志的CSV文件

pythonCopy Code
import csv

# 统计IP访问频次
ip_counter = {}
with open('access.log.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ip = row['client_ip']
        ip_counter[ip] = ip_counter.get(ip, 0) + 1

# 输出TOP10访问IP
sorted_ips = sorted(ip_counter.items(), key=lambda x: x, reverse=True)[:10]
for ip, count in sorted_ips:
    print(f"{ip}: {count}次")


pythonCopy Code
# 使用生成器逐块处理
def process_large_file(filename, chunk_size=10000):
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            chunk = [next(f).strip() for _ in range(chunk_size)]
            if not chunk:
                break
            # 在此处理数据块（如正则匹配）
            yield [line for line in chunk if 'ERROR' in line]

# 使用示例
error_lines = []
for partial_result in process_large_file('app.log'):
    error_lines.extend(partial_result)
print(f"发现{len(error_lines)}条错误日志")



Python内建数据类型有哪些‌：

    int、bool、str、list、tuple、dict‌1

‌简述with方法打开文件的作用‌：

    with方法用于自动管理资源，如文件，确保文件在使用后正确关闭，避免资源泄露‌1

‌Python中的可变数据类型和不可变数据类型‌：

    可变类型（mutable）：如list、dict，值可以改变而不会新建对象‌13
    不可变类型（immutable）：如str、tuple，值不能改变，改变会生成新对象‌

  Python进阶‌

    ‌Python多线程的限制‌：
        由于全局解释器锁（GIL），Python多线程在CPU密集型任务上可能不如多进程高效‌

    ‌Python多线程与多进程的区别‌：
        多线程共享进程内存空间，易于共享资源但受GIL限制；多进程有独立内存空间，避免资源竞争但需通过IPC通信‌

    ‌Python中的装饰器是什么‌：
        装饰器是修改其他函数行为的特殊函数，不修改原函数源代码，使用@符号加装饰器函数名‌

    ‌Python中的生成器是什么‌：
        生成器是特殊的函数，用于迭代生成一系列值，节省内存，使用yield语句返回值‌

‌Python应用与特性‌

    ‌Python的优点‌：
        简单易学、开发效率高、库丰富、跨平台、社区活跃‌35

    ‌Python的应用领域‌：
        Web开发、数据分析、人工智能、机器学习、桌面GUI开发等‌5

  

当然，以下是一些额外的Python进阶面试题，旨在进一步考察应聘者对Python语言及其应用的深入理解和应用能力：
1. ‌什么是上下文管理器（Context Manager）？请给出一个自定义上下文管理器的例子。‌

‌答案‌：
上下文管理器是一种对象，它定义了__enter__()和__exit__()方法，允许你在进入和退出代码块时执行特定操作，如资源获取和释放、异常处理等。以下是一个自定义上下文管理器的例子，用于管理文件的打开和关闭：

pythonCopy Code
class FileManager:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_type:
            return False  # 表示不抑制异常

# 使用上下文管理器
with FileManager('example.txt') as file:
    data = file.read()
    print(data)

2. ‌解释Python中的协程（Coroutine）和异步编程（Asyncio）。‌

‌答案‌：
协程是一种用户级的轻量级线程，它允许函数在执行过程中暂停和恢复，而不会阻塞整个程序。Python 3.5引入了async和await关键字来支持异步编程，使得协程的实现更加简洁和直观。asyncio是Python标准库中的一个模块，提供了编写单线程并发代码的基础设施，使用事件循环来调度协程的执行。
3. ‌请谈谈Python中的类型提示（Type Hints）及其重要性。‌

‌答案‌：
类型提示是Python 3.5引入的一种语法，用于为变量和函数参数指定类型。虽然类型提示不会影响程序的运行，但它们对于代码的可读性、可维护性以及静态类型检查工具（如mypy）的支持至关重要。通过类型提示，开发者可以更容易地理解代码的功能和预期输入/输出，从而减少错误并提高代码质量。
4. ‌如何实现一个单例模式（Singleton Pattern）在Python中？‌

‌答案‌：
单例模式是一种创建型设计模式，确保一个类只有一个实例，并提供一个全局访问点。在Python中，可以通过多种方式实现单例模式，如使用类属性、装饰器或元类等。以下是一个使用类属性实现单例模式的例子：

pythonCopy Code
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

# 使用单例模式
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # 输出: True

5. ‌Python中如何进行元编程（Metaprogramming）？请给出一个例子。‌

‌答案‌：
元编程是指在程序运行时创建或修改程序结构的行为。在Python中，元编程通常通过装饰器、元类、属性访问器等技术实现。以下是一个使用元类来修改类属性的例子：

pythonCopy Code
class Meta(type):
    def __new__(cls, name, bases, dct):
        new_dct = {}
        for attr, value in dct.items():
            if callable(value):
                new_dct[f'_{attr}'] = value
                def wrapper(self, *args, **kwargs):
                    print(f"Calling {attr}")
                    return getattr(self, f'_{attr}')(*args, **kwargs)
                new_dct[attr] = wrapper
        return type.__new__(cls, name, bases, new_dct)

class MyClass(metaclass=Meta):
    def my_method(self):
        return "Hello, World!"

# 使用元类修改后的类
obj = MyClass()
obj.my_method()  # 输出: Calling my_method, Hello, World!




以下是一些Python进阶面试题及其详细答案，这些问题旨在考察应聘者对Python语言的深入理解和应用能力：
1. 谈谈你对Python中GIL（全局解释器锁）的理解，以及它对多线程编程有什么影响？

‌答案‌：
GIL（Global Interpreter Lock，全局解释器锁）是Python解释器中的一个机制，用于确保在任何给定时间只有一个线程在执行Python字节码。这意味着即使存在多个线程，它们也不能并行执行CPU密集型任务。GIL的存在使得Python在处理I/O密集型任务时效率较高，但对于CPU密集型任务可能导致性能下降。因此，在Python中进行多线程编程时，需要考虑到GIL的影响，并可能需要采用其他方法（如多进程）来提高性能。
2. 请解释Python中的装饰器，并给出一个使用装饰器的例子。

‌答案‌：
装饰器是Python中的一个强大功能，它允许开发者在不修改原始函数代码的情况下，为函数提供额外的功能。装饰器本质上是一个接受函数作为参数的可调用对象（函数或类），并返回一个新的函数对象。以下是一个使用装饰器来记录函数执行时间的例子：

pythonCopy Code
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

@timer
def my_function():
    time.sleep(2)  # 模拟耗时操作

my_function()

3. 谈谈Python中的内存管理机制。

‌答案‌：
Python中的内存管理由Python内存管理器处理。它使用引用计数和垃圾回收机制来管理内存。当一个对象的引用计数降为0时，Python解释器会自动回收该对象的内存。此外，Python还提供了一个垃圾回收器来处理循环引用的情况，确保无用的对象能够被及时回收。了解Python的内存管理机制有助于开发者编写更加高效和健壮的代码。
4. 如何在Python中高效地处理大数据集？

‌答案‌：
在处理大数据集时，可以考虑以下几种方法来提高效率：

    使用生成器而不是列表推导或循环来逐个生成数据，以避免一次性加载整个数据集到内存中。
    使用NumPy、Pandas等库来处理数据，这些库针对大规模数据处理进行了优化。
    使用数据库来存储和管理数据，通过SQL查询来高效地获取和处理所需的数据。
    利用并行处理或多线程/多进程技术来加速数据处理过程（注意GIL的影响）。

5. 请解释Python中的异常处理机制，并给出一个使用try-except-else-finally块的例子。

‌答案‌：
Python中的异常处理机制用于捕获和处理程序中可能出现的异常。它使用try-except-else-finally块来实现。try块包含可能会引发异常的代码，except块用于捕获并处理异常，else块在try块成功执行且没有异常发生时执行，finally块无论是否发生异常都会执行，通常用于清理资源或释放锁等。以下是一个例子：

pythonCopy Code
try:
    # 可能会引发异常的代码
    result = 10 / 0
except ZeroDivisionError:
    # 处理除零异常的代码
    print("Cannot divide by zero")
else:
    # 如果没有异常发生，则执行此代码块
    print("Calculation was successful:", result)
finally:
    # 无论是否发生异常，都会执行此代码块
    print("Execution of try-except block is complete")




. 什么是字典和列表推导？

‌回答‌：
字典和列表推导是Python中的语法糖结构，它们允许你从给定的列表、字典或集合构建经过修改和过滤的列表、字典或集合。这些推导式可以节省大量时间和代码，使代码更加简洁和高效。例如：

    列表推导：squared_list = [x**2 for x in my_list]
    字典推导：squared_dict = {x: x**2 for x in my_list}

2. Python中的作用域解析是什么？

‌回答‌：
作用域解析是Python中用于确定变量访问权限的机制。Python中的变量作用域分为本地作用域、嵌套作用域、全局作用域和内置作用域。在解析变量时，Python会从当前作用域开始，逐层向外查找，直到找到变量为止。如果找不到，则会抛出NameError异常。
3. Python中的装饰器是什么，以及它们有什么用途？

‌回答‌：
装饰器是Python中用于修改其他函数或类的行为的函数。它们接受一个函数或类作为参数，并返回一个新的函数或类。装饰器常用于日志记录、性能分析、权限验证等场景。通过装饰器，可以在不修改原始函数或类代码的情况下，为其添加额外的功能。
4. Python中的生成器是什么，它们与列表有什么区别？

‌回答‌：
生成器是迭代器的一种，它允许你按需生成序列中的元素，而不是一次性生成整个序列。与列表相比，生成器更节省内存，因为它们在内存中只保存算法状态，而不是整个序列。生成器使用yield关键字来暂停和恢复函数的执行，从而可以逐个产生元素。
5. 如何实现一个斐波那契数列生成器？

‌回答‌：
可以使用生成器来实现斐波那契数列的生成。以下是一个简单的示例：

pythonCopy Code
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 使用生成器
fib = fibonacci()
for _ in range(10):
    print(next(fib))

6. Python中的内存管理机制是怎样的？

‌回答‌：
Python中的内存管理由Python内存管理器处理。它负责分配和释放内存。当对象的引用计数变为0时，Python会自动回收该对象的内存。此外，Python还内置了一个垃圾收集器，可以回收未使用的内存并释放给堆空间。
7. 如何优化Python代码的性能？

‌回答‌：
优化Python代码的性能可以从多个方面入手，包括但不限于：

    使用更高效的数据结构和算法
    避免不必要的全局变量和嵌套循环
    使用生成器而不是列表推导来处理大数据集
    利用多线程或多进程来并行处理任务
    使用C扩展模块或Cython等工具来优化关键部分的性能

请注意，以上只是部分可能的Python进阶面试题及其简要解答。实际面试中，面试官可能会根据应聘者的背景和经验提出更具针对性的问题。因此，在准备面试时，建议结合自身的实际情况进行有针对性的复习和练习。


    Python中的数据类型有哪些？‌
        Python中的数据类型包括整数（int）、浮点数（float）、字符串（str）、布尔值（bool）、列表（list）、元组（tuple）、字典（dict）、集合（set）以及NoneType等。

    ‌解释Python中的if __name__ == "__main__":语句的作用。‌
        这条语句用于判断当前脚本是否作为主程序运行。如果是，则执行其下的代码块。这对于模块化和测试代码非常有用。

    ‌Python中的异常处理是如何工作的？‌
        使用try、except和finally关键字。try块中的代码如果抛出异常，则会被except块捕获并处理。finally块中的代码无论是否发生异常都会执行。

数据结构

    ‌列表和元组的区别是什么？‌
        列表是可变的，即其元素可以修改、添加或删除；而元组是不可变的，一旦创建就不能修改。

    ‌字典的键值对是如何存储的？‌
        字典通过哈希表来存储键值对，键必须是不可变的（如字符串、数字或元组），而值可以是任何类型。

    ‌集合和列表的区别是什么？‌
        集合是一个无序的、不包含重复元素的数据结构；而列表是有序的，可以包含重复元素。

算法与逻辑

    ‌如何实现一个简单的排序算法（如冒泡排序）？‌
        通过多次遍历列表，比较相邻元素并交换它们的位置，直到列表完全有序。

    ‌解释时间复杂度和空间复杂度的概念。‌
        时间复杂度是算法执行所需时间的度量，通常表示为输入大小的函数；空间复杂度是算法执行所需存储空间的度量。

库与框架

    ‌你使用过哪些Python库？请举例说明。‌
        例如，NumPy用于数值计算，Pandas用于数据处理和分析，Matplotlib用于绘图和可视化，Django和Flask是Web框架等。

    ‌解释Django框架中的MTV模式。‌
        MTV模式即模型（Model）、模板（Template）和视图（View）。模型定义数据库结构，视图处理用户请求并返回响应，模板用于渲染HTML页面。

项目经验

    ‌描述一个你使用Python完成的项目。‌
        简要介绍项目背景、你负责的部分、使用的技术和遇到的挑战及解决方案。

    ‌你在项目中遇到过哪些性能问题？你是如何解决的？‌
        例如，通过优化算法、使用更高效的数据结构、缓存结果或并行处理等方式来提高性能。

准备面试时，除了复习上述知识点外，还要确保对简历中提到的项目和技能有深入的了解和准备。同时，练习编写代码和解释代码逻辑也是非常重要的。祝你面试成功！


如何在一个函数内部修改全局变量

利用global 修改全局变量

what is this lib for:   re:   正则匹配

字典如何删除键和合并两个字典
del和update方法

python的GIL
GIL 是python的全局解释器锁，同一进程中假如有多个线程运行，一个线程在运行python程序的时候会霸占python解释器（加了一把锁即GIL），使该进程内的其他线程无法运行，等该线程运行完后其他线程才能运行。如果线程运行过程中遇到耗时操作，则解释器锁解开，使其他线程运行。所以在多线程中，线程的运行仍是有先后顺序的，并不是同时进行。

多进程中因为每个进程都能被系统分配资源，相当于每个进程有了一个python解释器，所以多进程可以实现多个进程的同时运行，缺点是进程系统资源开销大


python实现列表去重的方法
先通过集合去重，在转列表


un(*args,**kwargs)中的*args,**kwargs什么意思？ **kwargs 不定长度的键值对


简述面向对象中__new__和__init__区别
__init__是初始化方法，创建对象后，就立刻被默认调用了，可接收参数

__new__至少要有一个参数cls，代表当前类，此参数在实例化时由Python解释器自动识别
__new__必须要有返回值，返回实例化出来的实例，这点在自己实现__new__时要特别注意，可以return父类（通过super(当前类名, cls)）__new__出来的实例，或者直接是object的__new__出来的实例

linux的tree 命令？


21、列出python中可变数据类型和不可变数据类型，并简述原理

不可变数据类型：数值型、字符串型string和元组tuple

不允许变量的值发生变化，如果改变了变量的值，相当于是新建了一个对象，而对于相同的值的对象，在内存中则只有一个对象（一个地址），如下图用id()方法可以打印对象的id

可变数据类型：列表list和字典dict；

允许变量的值发生变化，即如果对变量进行append、+=等这种操作后，只是改变了变量的值，而不会新建一个对象，变量引用的对象的地址也不会变化，不过对于相同的值的不同对象，在内存中则会存在不同的对象，即每个对象都有自己的地址，相当于内存中对于同值的对象保存了多份，这里不存在引用计数，是实实在在的对象。
