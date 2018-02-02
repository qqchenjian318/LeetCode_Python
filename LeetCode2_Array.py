# Valid Sudoku
# Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules
# http://sudoku.com.au/TheRules.aspx .
# The Sudoku board could be partially filled, where empty cells are filled with the character '.'.
# 填充数独里面的数字，空单元格用 ',' 填充
# 为了便于计算 我直接改成了0
# 5 3 0  0 7 0  0 0 0
# 6 0 0  1 9 5  0 0 0
# 0 9 8  0 0 0  0 6 0
#
# 8 0 0  0 6 0  0 0 3
# 4 0 0  8 0 3  0 0 1
# 7 0 0  0 2 0  0 0 6
#
# 0 6 0  0 0 0  2 8 0
# 0 0 0  4 1 9  0 0 5
# 0 0 0  0 8 0  0 7 9
#
# 切 我还以为是数独的解法
# 其实这个题只是让你判断是否是一个有效的数独
# 意思就是 已有的数据里面 横竖 小方块是否有重复的数据
# 而不需要填充数据
# 用一个长度为9的boolean数组
def valid_sudoku(A=[]):
    # 检查竖
    for i in range(9):
        # 重置判读数组
        B = [False] * 9
        for j in range(9):
            if not check_valid(B, A[i][j]):
                return False
    # 检查横
    for i in range(9):
        # 重置判断数组
        B = [False] * 9
        for j in range(9):
            if not check_valid(B, A[j][i]):
                return False
    # 检查小方块
    for i in range(3):
        for j in range(3):
            B = [False] * 9
            for a in range(3):
                for b in range(3):
                    x = i * 3 + a
                    y = j * 3 + b
                    if not check_valid(B, A[x][y]):
                        return False
    return True


# 检查某个点的有效性
def check_valid(A=[], i=0):
    if i == 0:
        return True
    if i < 1 or i > 9 or A[i - 1]:
        return False
    A[i - 1] = True
    return True


A = [[5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
     [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
     [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9]]


# Trapping Rain Water
# Given n non-negative integers representing an elevation map where the width of each bar is 1, compute
# how much water it is able to trap after raining.
# For example, Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
# 雨水收集器
# 网上有图 看着更清楚 (建议自己百度一下)
# 基本思路就是 每根柱子 如果左边或者右边 没有比他高的数  那么就不能存储雨水
# 如果左右两边都有比他高的数  那么 就用左右两边中小的那个 减去他本身 就可以得到本柱子能存储多少雨水了
# 然后再进行叠加即可
#

def trap_rain_water(L=[]):
    result = 0
    for i in range(len(L)):
        left = 0
        right = 0
        x = i
        y = i
        while x > 0:
            x -= 1
            if L[x] > L[i] and L[x] > left:
                left = L[x]
        while y < len(L) - 1:
            y += 1
            # print(y)
            if L[y] > L[i] and L[y] > right:
                right = L[y]
        if left == 0 or right == 0:
            continue
        result += (min(left, right) - L[i])
    print('这个模型 应该能收集的雨水是 %s  ' % result)


# 该题的第二种思路
# 先遍历一遍 找到最高点 （有多个最高点 就取第一个）
# 然后将数组分成两边 分别依次处理 左右两边即可
# 就不写代码了 思路很简单
def trap_rain_water_2(L=[]):
    pass


# Rotate Image
# You are given an n * n 2D matrix representing an image.
# Rotate the image by 90 degrees (clockwise).
# Follow up: Could you do this in-place?
# 有一个 n * n 的2D图片 让这个图片旋转90度
# 比如一个二维数组
# 3 3 3
# 2 1 5
# 8 4 2
#
# 旋转90度之后
# 8 2 3
# 4 1 3
# 2 5 3
# 思路有很多种
# 对角线旋转
# 首先以对角线为轴 翻转
# 然后再以x轴中线上下翻转即可得到结果

def rotate_image(L=[]):
    s = len(L)
    print(L)
    # 对切线交换
    for i in range(s):
        for j in range(s - 1):
            if L[i][j] == L[s - j - 1][s - i - 1]:
                # 如果等于的话 说明到了中线了 就不需要进行遍历这一行了 开始下一行
                break
            L[i][j], L[s - j - 1][s - i - 1] = swap(L[i][j], L[s - j - 1][s - i - 1])
    print(L)
    # x轴的中线交换
    for i in range(s // 2):
        for j in range(s):
            L[i][j], L[s - i - 1][j] = swap(L[i][j], L[s - i - 1][j])

    print(L)


def swap(x=0, y=0):
    return y, x


# Plus One
# Given a number represented as an array of digits, plus one to the number.
# 对一个数组进行加1的操作
# 如果大于 10 就需要进位
# 如果最高位了还需要进位的话 就重新new一个数组
# 并且将最高位设为1 其余为设为0
# 这题可能会涉及到很多扩展 比如两个数组相加
#
def plus_one(L=[], a=1):
    s = len(L)
    z = a
    for i in range(s):
        L[i] += z
        z = L[i] // 10
        L[i] = L[i] % 10
    if z > 0:
        # 说明需要进位
        L.insert(s, 1)
    return L


# Climbing Stairs
# You are climbing a stair case. It takes n steps to reach to the top.
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# 爬楼梯 一共了n阶 每次可以爬1阶或者2阶 问一共有多少种上楼的方式

# 上第n阶的可能有 1步上来的 或者2步上来的 那么
# 上n阶的可能的路线就是 上 n - 1阶的路线 + n - 2阶的路线
# 所以 f(n) = f(n - 1) + f(n - 2)
# 递归的方式 如下


def climbing_stairs(n=0):
    if n == 1:
        return 1
    if n == 2:
        return 2
    return climbing_stairs(n - 1) + climbing_stairs(n - 2)


# 其实这道题 如果用遍历的方式也可以解决的
# 其实就是一个斐波那契数列，因为f(n) = f(n - 1) + f(n - 2)
# 而 n = 1的时候 解 = 1
# n = 2的时候  解 = 2
# n = 3 的时候 解 = 1 + 2 = 3
# 1 1 2 3 5 8


def climbing_stairs2(n=0):
    i = 1
    cur = 1
    prev = 0
    while i <= n:
        temp = cur
        cur += prev
        prev = temp
        i += 1
    return cur


# Gray Code
# The gray code is a binary numeral system where two successive values differ in only one bit.
# Given a non-negative integer n representing the total number of bits in the code, print the sequence of
# gray code. A gray code sequence must begin with 0.
# For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:
# 00 - 0
# 01 - 1
# 11 - 3
# 10 - 2
# Note:
# • For a given n, a gray code sequence is not uniquely defined.
# • For example, [0,2,3,1] is also a valid gray code sequence according to the above definition.
# • For now, the judge is able to judge based on one instance of gray code sequence. Sorry about that.

# 格雷编码
# 格雷编码是一个二进制数字系统，其中两个连续的值只有一个比特不同
# 给定一个非负整数 打印这个数的格雷序列编码，一个格雷码必须以0开头
# 例如 给n = 2 ，返回的是[0,1,3,2]

# gray code
# 格雷码 的计算方式是 保留自然二进制码的最高位作为格雷码的最高位，格雷码次高位为二进制码的高位与次高位异或
# 其余各位的求法类似
# 例如 自然二进制码 1001 ，转换成格雷码的过程是：保留最高位，然后将第1位的1和第2位的0异或，得到1，作为格雷码的第2位
# 将第2位的0和第3位的0异或，得到0，作为格雷码的第3位
# 将第3位的0和第4位的1异或，得到1，作为第4位，最终，格雷码为1101

# 自然二进制码转换为格雷码
# 保留格雷码的最高位作为自然二进制码的最高位，次高位为自然二进制高位和格雷码次高位异或，其余各位类似
# 比如讲格雷码1000 转换为自然二进制码的过程是：
# 保留最高位1，作为自然二进制码的最高位，然后将自然二进制码的第1位的1和格雷码的第2位的0异或，得到1，作为自然二进制码的
# 第2位，将自然二进制码的第3位1和格雷码的第4位9异或，得到1，作为自然二进制码的第4位，最终，自然二进制码为1111
#
# 异或：如果两个值不相同，则异或的结果为1，如果两个值相同，则异或的结果为0
# 这道题要求生成 n 位二进制代码的所有格雷码
# 比如 n = 3
# 二进制 》 格雷码 》 10进制
# 000 》 000 》 0
# 001 》 001 》 1
# 010 》 011 》 3
# 011 》 010 》 2
# 100 》 110 》 6
# 101 》 111 》 7
# 110 》 101 》 5
# 111 》 100 》 4

# 利用数学公式，对0 到 2~n - 1的所有整数，转化为格雷码
#
def gray_code(n=0):
    size = 1 << n
    result = list()
    for i in range(size):
        result.append(binary_to_gray(i))
    return result


def binary_to_gray(i=0):
    return i ^ (i >> 1)  # 当前位的值 异或上 上一位的值


# Set Matrix Zeroes
# Given a m * n matrix, if an element is 0, set its entire row and column to 0. Do it in place.
# Follow up: Did you use extra space?
# A straight forward solution using O(mn) space is probably a bad idea.
# A simple improvement uses O(m + n) space, but still not the best solution.
# Could you devise a constant space solution?
#  给定一个 m * n的矩阵，如果元素是0 ，那么就将整个行和列设置为0，
# 你是否使用了额外的内存空间
# 如果解法使用了O(m * n)大小的内存，就是一个bad idea
# 如果使用的是O(m + n)大小的内存，就有一些进步了，但是依然不是最好的解法
# 你能否实现一个使用常数 大小的内存空间的解法呢？

def matrix_zeroes(L=[]):
    m = len(L)
    n = len(L[0])
    print(L)

    row_has = False
    column_has = False
    # 说明行有0
    for i in range(m):
        if L[i][0] == 0:
            row_has = True
            break
    # 说明竖有0
    for j in range(n):
        if L[0][j] == 0:
            column_has = True
            break

    # 遍历其他行列  如果有0 就将该数组的值设为0
    for i in range(m):
        for j in range(n):
            if L[i][j] == 0:
                L[i][0] = 0
                L[0][j] = 0

    # 分别数组 如果第一行和第一列对应位置为0 那么设置为0
    for i in range(m):
        for j in range(n):
            if L[i][0] == 0 or L[0][j] == 0:
                L[i][j] = 0
    if row_has:
        for i in range(m):
            L[i][0] = 0
    if column_has:
        for j in range(n):
            L[0][j] = 0

    print(L)


# Gas Station
# There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
# You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next
# station (i+1). You begin the journey with an empty tank at one of the gas stations.
# Return the starting gas station’s index if you can travel around the circuit once, otherwise return -1.
# Note: The solution is guaranteed to be unique.
#
# 沿途有n个加气站 ，每个站拥有的天然气数量是 gas[i]，你有一辆 无限大气罐的车，到下一个加气站(i + 1)需要花费的天然气 是 costs[i]
# 你从目前是空车 并从任意一个加气站出发
# 如果能够环绕一圈 就返回你出发的加气站的index ，否则的话 返回 -1
#
# 额。。。这道题呢 主要就是一个关键点 消耗的天然气和加油站拥有的天然气的差值
# 比如可以从差值最大的地方出发
#
# 每个站点的结余可以看着 diff[i] = gas[i] - cost[i]
# 那么如果从m点出发 到达p点 车里剩余的天然气数量是 diff[m] + diff[m+1]+....+diff[p] = sum1 ，如果sum1 < 0 ，说明从m点出发不能环绕一圈
# 那么我们从 p+1点出发，直到n点 剩余的天然气数量是 diff[p+1]+diff[p+1+1]+....+diff[n] = sum2，如果sum2 <0 ，说明还是不能成功环绕一圈
# 那么我们从 n+1点出发，知道m点，剩余的天然气数量是 diff[n+1] + diff[n+1+1]+...+diff[m] = sum3 ，如果sum3 > 0 ,说明从n+1点到m点，还能有剩余的天然气
# 那么依次类推 如果 sum1 + sum2 +...+sumN > 0 说明就可以环绕一圈
# 那如果不行的话 就不可以
# 所以

def gas_station(gas=[], cost=[]):
    total = 0
    sum1 = 0
    index = 0
    i = 0
    while i < len(gas):
        total += gas[index] - cost[index]
        sum1 += gas[index] - cost[index]
        # 当前的所有值 如果total 小于0 那么就重置sum 并记下标记点
        if sum1 < 0:
            sum1 = 0
            index = i

    if total > 0:
        return index + 1
    return -1


# Candy
# There are N children standing in a line. Each child is assigned a rating value.
# You are giving candies to these children subjected to the following requirements:
# • Each child must have at least one candy.
# • Children with a higher rating get more candies than their neighbors.
# What is the minimum candies you must give?
#
# 这里有n个小孩子站在一条线上，每个都有一个评分值
# 你按照下面的要求 给这些小孩子分糖果
# 每个孩子至少有一个糖果
# 如果孩子的评分值越高 他获得的糖果就比他临近的人更多
# 请问  至少需要多少个糖果
#
# 判断两个分值大小
# 如果A[i+1] > A[i] 那么 A[i+1]的糖果最好的数量就是 i位置的糖果数量 + 1
# 那如果A[i+1] <= A[i] 那应该给多少糖果呢？好像不太能确定到底该给多少糖果 因为我们需要给尽可能的少
#
# 我们的思路就可以 首先遍历一遍 大的话 就+1
# 如果小的话 我们就 不知道应该填入多少了
# 如果遇到 1 5 4 3 2 这样的数列
# 首先 1孩子 给1个糖果
# 5孩子给 2个糖果的话
# 那么 4孩子 只能给一个糖果
# 那么3孩子就不能给了 所以就有问题
# 说明5孩子应该给的比预计中的还要多
# 而这个5孩子给的数量  其实跟 5 4 3 2 这个递减数列的长度有关系
# 如果递减数列 长度为4  那么第一个值至少也要为4
# 所以 我们可以通过引入一个修正值的方式 来实现一次遍历 就计算出total数
# 修正值的作用就是

# 思路：
# 首先 从左往右遍历 确保 右边比左边高的小朋友的糖果数比左边多
# 然后 从右往左遍历 使得 左边比右边得分高的小朋友 糖果数比右边多
#
# 例如 1 5 4 3 2
# 第一次遍历
# 1 2 0 0 0
# 第二次遍历
# 1 4 3 2 1

# 例如 1 5 4 2 3
# 第一次遍历
# 1 2 0 0 1
# 第二次
# 1 3 2 1 2

# 当然 题目要求 只需要total 也就是最少的糖果数量
# 并不需要知道每个孩子具体分多少个
# 所以 其实还有更加节省空间和时间的方法
# 这里就不多说了
#
def candy(L=[]):
    result = list(range(len(L)))
    print(L)
    for i in range(len(L)):
        if i == 0:
            result[i] = 1
            continue
        if L[i] > L[i - 1]:
            if result[i - 1] == 0:
                result[i - 1] = 1
            result[i] = result[i - 1] + 1
        else:
            result[i] = 0

    j = len(L) - 1
    while j > 0:
        if j == len(L) - 1 and result[j] == 0:
            result[j] = 1
            j -= 1
            continue
        if L[j] > L[j + 1]:
            if result[j] <= result[j+1]:
                result[j] = result[j+1] + 1
        j -= 1
    print(result)


# Single Number
# Given an array of integers, every element appears twice except for one. Find that single one.
# Note: Your algorithm should have a linear runtime complexity. Could you implement it without using
# extra memory?
#
# 一个数异或同一个数两次的话 就是他本身
# 所以 我们可以根据异或的特点 写出时间复杂度O(n)和空间复杂度O(1)的算法

def single_num(L=[]):
    result = 0
    for i in range(len(L)):
        result = result ^ L[i]

    return result

# Single Number II
# Given an array of integers, every element appears three times except for one. Find that single one.
# Note: Your algorithm should have a linear runtime complexity. Could you implement it without using
# extra memory?
#
# 有一个int数组，除了某一个元素之外，每个元素都出现了三次 ，找出只出现了一次的这个，而且要求不能使用额外的内存空间
# 这个是上面那个题的变种

# ~按位取反
# ^ 异或

def single_sum_2(L=[]):
    a = 0
    b = 0
    for i in range(len(L)):
        b = (b ^ L[i]) & ~a
        a = (a ^ L[i]) & ~b

    return b

print(single_sum_2([1, 5, 1, 1, 2, 2, 2]))
