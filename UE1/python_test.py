#range(start, stop[, step])可创建一个整数列表
# start: 计数从 start 开始。默认是从 0 开始。例如range（5）等价于range（0， 5）;
# stop: 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
# step：步长，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)

# a=6
# b=2
# for i in range(a):
#     if b != i:
#         print("b != {}".format(i))
#     else:
#         print("b = {}".format(i))
#
#     print('Hallo zusammen')
# print("b != {}".format(2))
# print("网站名：{name}, 地址 {url}".format(name="菜鸟教程", url="www.runoob.com"))
# # 通过字典设置参数
# site = {"name": "菜鸟教程", "url": "www.runoob.com"}
# print("网站名：{name}, 地址 {url}".format(**site))
#
# # 通过列表索引设置参数
# my_list = ['菜鸟教程', 'www.runoob.com']
# print("网站名：{0[0]}, 地址 {0[1]}".format(my_list))  # "0" 是必须的

# row = ['00006.ppm','926','350','989','414','2']
# variable1 = int(row[0].split('.')[0])

# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
# 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
# dict() 函数用于创建一个字典。
# list1 = ['A','B','C','D','D','E','E','F',1]
# list2 = [1,2,3,4,4,5,6,6,6]
# zipped = zip(list1,list2) # 打包为元组的列表,元素个数与最短的列表一致
# my_dict = dict(zip(list1,list2))
# print(my_dict)
# zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
import numpy

# y = numpy.array([[[0,1,2,3],[0,4,5,6],[0,7,8,9]]])
# print(y[0][1])

# list = list(map(int, '2'))
# print(list)
values = ["a", "b", "c"]
for count, value in enumerate(values):
    print(count, value)