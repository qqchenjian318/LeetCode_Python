# coding=utf-8
# Given a sorted array, remove the duplicates in place such that each element appear only once
# and return the new length.
# Do not allocate extra space for another array, you must do this in place with constant memory.
# For example, Given input array A = [1,1,2],
# Your function should return length = 2, and A is now [1,2].
from math import factorial
from re import L

import sys


def remove_duplicates(L=[]):
    x = 0
    n = 1
    index = 0
    while n < len(L):
        if L[x] != L[n]:
            x += 1
            L[x] = L[n]
            index += 1
        n += 1

    return L[0:index + 1]


# Follow up for ”Remove Duplicates”: What if duplicates are allowed at most twice?
# For example, Given sorted array A = [1,1,1,2,2,3],
# Your function should return length = 5, and A is now [1,1,2,2,3]
def remove_duplicates_twice(L=[]):
    x = 0
    n = 1
    index = 0
    count = 0
    while n < len(L):
        if L[n] != L[index]:
            x += 1
            L[x] = L[n]
            count = 0
            index += 1
        else:
            if count <= 1:
                x += 1
                L[x] = L[n]
                count += 2
                index += 1

        n += 1
    return L[0:index + 1]


# Suppose a sorted array is rotated at some pivot unknown to you beforehand.
# (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
# You are given a target value to search. If found in the array return its index, otherwise return -1.
# You may assume no duplicate exists in the array.
# 不管旋转还是没有旋转过 每次二分 总有一边是有序的
def search_in_rotated_array(L=[], target=0):
    last = len(L)
    first = 0
    while first != last:
        mid = int((first + (last - first) / 2))
        if L[mid] == target:
            return mid
        if L[first] <= L[mid]:
            # 说明 这边是有序的
            if L[first] <= target <= L[mid]:
                last = mid
            else:
                # 说明 在另一边
                first = mid + 1
        else:
            if L[mid] <= target <= L[last - 1]:
                first = mid + 1
            else:
                last = mid
    return -1


# Follow up for ”Search in Rotated Sorted Array”: What if duplicates are allowed?
# Would this affect the run-time complexity? How and why?
# Write a function to determine if a given target is in the array.
# 这个跟上上面的有一个不同点就是，可能会出现等于的情况 而且如果目标存在重复的
# 那么可能只会随机返回其中一个的index

def search_in_rotated_duplicates_array(L=[], target=0):
    last = len(L)
    first = 0
    while first != last:
        mid = int(first + (last - first) / 2)
        if L[mid] == target:
            return mid
        if L[first] < L[mid]:
            # 说明是有序的
            if L[first] <= target <= L[mid]:
                last = mid
            else:
                first = mid + 1
        elif L[first] > L[mid]:

            # 说明不是有序的 那么另一边有两种情况 刚好等于 以及是有序的
            if L[mid] <= target <= L[last - 1]:
                first = mid + 1
            else:
                last = mid
        else:
            # 说明第一个字和中间值相等 那么index 肯定不是第一个 所以 直接first++ 就好
            first += 1

    return -1


# There are two sorted arrays A and B of size m and n respectively. Find the median of the two sorted
# arrays. The overall run time complexity should be O(log(m + n)).
# 找到两个排序数组中的 中间值 median(中间的 )
# 該题更通用的说法应该是 从两个给定的排序数组中找到第k大的元素的位置
# 时间复杂度 要求O(log(m + n))
#
# 思路
# 1 两个数组进行合并依次读取数据直到第 k大的元素 但是时间复杂度三O(m + n)
# 2 如果是取中间的 那么 k = ( m + n) / 2
# 3 那么可以用二分的思想 一半一半的排除 才能让时间复杂度达到O(log ( m + n))
# 4 首先 将m数组的 第m/2个元素 和n数组的第 n/2个元素进行比较
# 5
# #
def find_k_from_two_sort_arrays(L=[], B=[], k=1):
    # k表示第k大的元素
    a = len(L)
    b = len(B)
    l = a + b

    pass


def find_kth(A=[], n=0, B=[], m=0, k=0):
    if m == 0:
        return B[k]

    if L[k] == B[k]:
        return L[k]
    elif L[k] < B[k]:
        pass
    else:
        pass


# Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
# For example, Given [100, 4, 200, 1, 3, 2], The longest consecutive elements sequence is [1,
# 2, 3, 4]. Return its length: 4.
# Your algorithm should run in O(n) complexity.
# 一个没有排序的数组，找出最长的连续数组的长度 时间复杂度在O(n)以内
# 意思就是一次读取数据 就完成选择 用一个dict来记录 所有数据 i
# 然后 依次i++ 和i-- 如果有数据 并且还没用过 就length++  直到所有的数据 都循环一遍
#
def get_sort_length(A=[]):
    d = {}
    for i in A:
        d[i] = False

    longest = 0
    for i in d:
        if d[i]:
            return
        length = 1
        d[i] = True
        z = i - 1
        while z >= 0:
            if d.get(z) is None:
                break
            else:
                if not d.get(z):
                    length = length + 1
            z = z - 1

        b = i + 1
        while b <= (len(d) - 1):

            if d.get(b) is None:
                break
            else:
                if not d.get(b):
                    length = length + 1
            b = b + 1
        if length > longest:
            longest = length

    return longest


# Given an array of integers, find two numbers such that they add up to a specific target number.
# The function twoSum should return indices of the two numbers such that they add up to the target, where
# index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not
# zero-based.
# You may assume that each input would have exactly one solution.
# Input: numbers={2, 7, 11, 15}, target=9
# Output: index1=1, index2=2
# 需要返回index值，所以
# 1、如果用双重for循环的话 时间复杂度是O(n~2) 会超时
# 2、先排序 然后二分 找到 小于 9的位置 但是要返回index下标，所以也不能进行排序
# 3、用一个dict  存储每个值的下标index
#
def two_sum(A=[], target=0):
    d = {}
    for i in range(len(A)):
        d[A[i]] = i
    for z in range(len(A)):
        gap = target - A[z]  # 算出和A[i] 匹配的数字的值，然后去map中寻找
        if d.get(gap) is not None and d[gap] > z:  # 因为是从0开始的，所以 后面的数字只用判断更后面的即可，所以加上>z的判定
            print('第一个值的index= %s，第二个字的index= %s' % (z, d[gap]))


# Given an array S of n integers, are there elements a; b; c in S such that a + b + c = 0? Find all unique
# triplets in the array which gives the sum of zero.
# Note:
# Elements in a triplet (a; b; c) must be in non-descending order. (ie, a  b  c)

# The solution set must not contain duplicate triplets.
# For example, given array S = {-1 0 1 2 -1 -4}.
# A solution set is:
# (-1, 0, 1)
# (-1, -1, 2)
# 从一个数组中找到 三个数 a，b，c ，并且 a + b + c = target
# 输出结果升序排列
# 输出结果没有重复的
#
# 我们需要注意的是，这个数据集 是无序的 而且有可能包含重复的值
# 输出的也并不是index而是值本身
# 所以，我们可以对源数据 先进行排序 快速排序的时间复杂度是O（nlgn）
# 然后先固定一个数 再找到其他符合标准的两个数
# 需要注意的有两点
# 1、如果固定的那个数的下一个数是重复的，那么需要跳过下一个数，不然会出现重复的式子
# 2、如果 x + y + z = target 的时候，j和k同时移动
# 3、 在j和k同时移动的时候 要判断 他们的下一个值是否和上一个值相同 如果都相同 说明是重复的数字
# 时间复杂度  貌似是O(n * n)
def three_sum(A=[], target=0):
    B = sorted(A)
    print(B)
    for i in range(len(B) - 2):
        # 先固定一个数 B[i]
        if i > 0 and B[i] == B[i - 1]:
            i += 1
            continue
        x = B[i]
        j = i + 1
        k = len(B) - 1
        print('外层循环---::%s' % x)
        while j < k:
            if x + B[j] + B[k] < target:
                j += 1
                while x + B[j] + B[k] < target:
                    j += 1
            elif x + B[j] + B[k] > target:
                k -= 1
                while x + B[j] + B[k] > target:
                    k -= 1
            else:
                print('满足要求的输出结果为  %s  %s  %s' % (x, B[j], B[k]))
                j += 1
                k -= 1
                # 当有满足要求的式子的时候 需要跳过附近重复的数字
                while B[j] == B[j - 1] and B[k] == B[k + 1] and j < k:
                    j += 1
                    k -= 1


# Given an array S of n integers, find three integers in S such that the sum is closest to a given number,
# target. Return the sum of the three integers. You may assume that each input would have exactly one solution.
# For example, given array S = {-1 2 1 -4}, and target = 1.
# The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
# 跟上面那道题很像 但是不同的是 不再判断是否 = target 而是记录 和target的差最小的
# 其实道理都是一样的

def three_sum_closest(A=[], target=0):
    B = sorted(A)
    print(B)
    value = 100
    ss = ''
    for i in range(len(B) - 2):
        x = B[i]
        if i > 0 and B[i] == B[i - 1]:
            continue
        j = i + 1
        k = len(B) - 1
        while j < k:
            result = abs((x + B[j] + B[k]) - target)
            # 计算出当前的差值
            print('resulet %s and value %s' % (result, value))
            if result < value:
                ss = '当前记录的大小 %s  %s %s ' % (x, B[j], B[k])
                value = result
            if x + B[j] + B[k] < target:
                j += 1
            else:
                k -= 1

    print(ss)


# Given an array S of n integers, are there elements a; b; c, and d in S such that a+b+c+d = target?
# Find all unique quadruplets in the array which gives the sum of target.
# Note:
# • Elements in a quadruplet (a; b; c; d) must be in non-descending order. (ie, a  b  c  d)
# • The solution set must not contain duplicate quadruplets.
# For example, given array S = {1 0 -1 0 -2 2}, and target = 0.
# A solution set is:
# (-1, 0, 0, 1)
# (-2, -1, 1, 2)
# (-2, 0, 0, 2)
# 跟上面的差不多是一个系列的问题
# 如果像上面那样  先排序 然后左右夹逼的话 时间复杂度会到达 O(n ~ 3) 会超时 所以 需要其他的优化策略
#
def four_sum_normal(A=[], target=0):
    # 对源数据 进行排序
    B = sorted(A)
    print(B)

    for i in range(len(B) - 3):
        j = i + 1
        X = B[i]
        while j <= len(B) - 2:
            Y = B[j]
            k = j + 1
            l = len(B) - 1
            while k < l:
                if X + Y + B[k] + B[l] < target:
                    k += 1
                elif X + Y + B[k] + B[l] > target:
                    l -= 1
                else:
                    print('%s  %s  %s  %s  ' % (X, Y, B[k], B[l]))
                    l -= 1
                    k += 1
            j += 1


# 优化后的4sum 答案
def four_sum_fast(A=[], target=0):
    B = sorted(A)
    print(B)
    twoSum = {}
    # 先用一个map集合 保存两位数的和 O( n ~ 2)
    for i in range(len(B) - 1):
        j = i + 1
        while j < len(B):
            twoSum[B[i] + B[j]] = [i, j]
            j += 1

    # 然后再左右夹逼
    for k in range(len(B)):
        l = k + 1
        while l < len(B):
            key = target - B[k] - B[l]
            s = twoSum.get(key)
            if s is not None and s[0] > k and s[0] > l and s[1] > k and s[1] > l:
                print('结果是：%s  %s  %s  %s ' % (B[k], B[l], B[s[0]], B[s[1]]))
            l += 1


# Given an array and a value, remove all instances of that value in place and return the new length.
# The order of elements can be changed. It doesn’t matter what you leave beyond the new length.
# 移除掉array里面的某个值，而且返回新的数组的长度
#
def array_remove(A=[], value=0):
    for i in A:
        if i == value:
            A.remove(i)

    print('结果是 %s  length %s' % (A, len(A)))


# 用index来实现数据移除
def array_remove_2(A=[], value=0):
    index = 0
    for i in range(len(A)):
        if A[i] != value:
            A[index] = A[i]
            index += 1
    print('结果是 %s  length %s' % (A[0: index], index))


# Implement next permutation, which rearranges numbers into the lexicographically next greater permutation
# of numbers.
# If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending
# order).
# The replacement must be in-place, do not allocate extra memory.
# Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the
# right-hand column.
# 1,2,3 → 1,3,2
# 3,2,1 → 1,2,3
# 1,1,5 → 1,5,1
# 题意就是寻找比当前排列顺序大的下一个排列
# 因为降序是没法变得更大的 所以 降序是从后往前找到第一个升序对的位置
#
# 思路是
# 从右向左 找到第一个降序的地方 即 A[i] > A[i - 1]
# 在i - 1 的右边 从右到左 找到 第一个 A[j] > A[i - 1]
# 将 j 和 i - 1交换 然后将 j右边的进行降序排列
# 要求不能增加额外的内存空间
# 1243
# 1324
# 为了图方便 使用了sorted函数 增加了额外的内存消耗。。。
#
def find_big_sort(A=[]):
    length = len(A)
    j = len(A)
    while j >= 0:
        j -= 1
        if j == 0:
            B = sorted(A[:length - 1])
            z = 0
            while z < len(B):
                A[z] = B[z]
                z += 1
        if A[j] > A[j - 1]:
            # 找到 A[i] > A[i - 1] 开始寻找第一个大于 A[i- 1]的数A[j]
            i = length
            while i > j - 1:
                i -= 1
                if A[i] > A[j - 1]:
                    # 交换i 和 j - 1
                    temple = A[i]
                    A[i] = A[j - 1]
                    A[j - 1] = temple
                    break
            # 升序排列 A[j] --A[length - 1]
            B = sorted(A[j:length], reverse=False)
            k = 0
            while k < len(B):
                A[j + k] = B[k]
                k += 1
            break
    return A


# The set [1,2,3,⋯,n] contains a total of n! unique permutations.
# By listing and labeling all of the permutations in order, We get the following sequence (ie, for n = 3):
# "123"
# "132"
# "213"
# "231"
# "312"
# "321"
# Given n and k, return the kth permutation sequence.
# Note: Given n will be between 1 and 9 inclusive.
# 其实这题跟上一题是一脉相承的 上题求的是 下一个更大的排列
# 那么这题也可以借用上题的代码
# 首先 我们初始化第一个排列 然后求下一个排列 直到求出 前k个排列
# 需要检查k的边界，因为要求是不重复的排列 那么n个数的排列 最多就 n的阶乘个

def permutation_seq_1(n=0, k=0):
    A = list(range(n + 1))
    a = 1
    while a <= n:
        A[a] = a
        a += 1
    print(A[1:])
    s = A[1:].__str__()
    B = A[1:]
    # 如果k大于 n的阶乘，也就是说 并没有这么多不重复排列的话 就重置k的值 为n个数 最多的排列方式
    if k > factorial(n):
        k = factorial(n)
    while k > 1:
        k -= 1
        B = find_big_sort(B)
        s = s + '  ' + B.__str__()

    return s


# 还是求1、2、...n 的前k个无重复排列
# 上面是暴力的方式 去求所有的排列 那么下面我们就利用康托的编码思路 进行优化
# 首先科普一下 什么是康托展开 X=a[n]*(n-1)!+a[n-1]*(n-2)!+...+a[i]*(i-1)!+...+a[1]*0!
# 所谓的康托展开 表示的是当前排列 在n个不同元素的全排列中的名次，比如213 在这3个数的所有排列中排第3
# 其中 a[i] 为当前未出现的元素中排在第几个（从0开始）(注意 这里是因为编码器的符合限制 所以才写出这样的 并不是传统的
# 表示 a数列中的第i个元素 而是 a[i]表示的是当前未出现的元素中排列在第几个)
# 怎么理解a[i]呢 我们可以通过举例来说明
# 比如 排列 312 他的康托展开就是
# X = a[3] * (3 - 1)! + a[1] * ( 1 - 1)! + a[2] * ( 2 - 1)!
# a[3] 此时i是3 而当前未出现的元素 就只有 1 2 ，那么如果从0 开始的话， 3 在 1 2 中的排列位置 就是 2 所以 a[3] = 2
# a[1] 此时i = 1 而当前未出现的元素 只有 2，那么1 的排列位置就是 0 所以 a[1] = 0
# a[2] 此时 i = 2 而当前未出现的就没有了 所以 他肯定排列位置是0 所以 a[2] = 0
# 所以 X = 2 * 2! + 0 * 0! + 0 * 1! = 4
# 意思就是 排列 312 ，在所有的这三个数的排列中 排列第 5 ，因为前面有4个比他小的排列
#  元素 1 2 3的全排列如下
#  [1, 2, 3]  [1, 3, 2]  [2, 1, 3]  [2, 3, 1]  [3, 1, 2]  [3, 2, 1]
#  可以看到 312 确实是排列在 第五个的
# 那么为什么康托展开 可以准确的计算出当前排列在所有排列中的位置呢？
# 我们来看 他的核心思想 其实就是 a[i] 表示在当前未出现的元素中排列第几
# 比如上面的例子 3 3前面如果出现过的元素 我们就不管了 那么在所有没出现的元素中 1 2 只要是比3小的数字  那么 如果把3的位置替换为
# 该数字 该数字后面的排列 都比 3在该位置的排列小
# 比如 312  如果把1放在第一位 那么存在的数列就是 123   132  他们肯定是比 312小的
# 如果把2 放在第一位那么 存在的数列 就是 213 231 他们肯定也是比 312 小的
# 所以 我们可以得出 所以未出现的元素 比i小的元素 他们与i交换位置 之后 所有的排列 都比原数列小
# 这是第i位的情况 那么依次类推 就可以得到康托展开式了 从而计算出所有比当前排列小的排列的数量

# 搞清楚了康托展开 那么跟我们这道题有什么关系呢？
# 我们要求的是 前k个排列
# 康托展开 是排列 到自然数的映射
# 那么他自然可以逆展开 从自然数 求得排列
# 那么我们的问题就变成了 求第k个排列 从而自然可以算出 第k -1 直到 第1个排列
# 那么怎么来求第k个排列呢？
# 比如 求3个元素的第4个排列
# 根据康托展开 4 - 1 = a[x]* (x-1)! + a[y] * (y - 1)! +  a[z] * (z - 1)
# 3 % 2! = 1 ~ 1
# 1 % 1! = 1 ~ 0
# 说明 第一位数 比x 小的数 有1 个 在123三个数中 那应该是2
# 第二位 比 y 小的数有1个  在未出现的元素13中 只有3 那么y = 3
# 第三位 那z = 1
# 那么结果应该是 213
#  3 1 2
# 所以 逆展开的思路其实就是
# 第一个数 a.1 = (k - 1) % (n - 1)! 为什么呢？是因为 如果你想要k这个值非常的大  那么 你第一个数肯定就得非常的大 n个数的全排列是 n ! 个
# 如果 k > (n - 1)! 那说明 第一位 肯定大于1 ，因为所有第一位为1的排列个数 也只有(n - 1)!个
# 那么 k / (n - 1)! = x 得到的x，其实就可以确定 第一个数 在所有未出现的元素中的位置了
# 比如 如果x = 3，那么说明 k 肯定是大于等于 3 * (n - 1) 小于 4 * (n - 1)! 那么第一个数就能确定了
# 依次类推 就可以确定排列中的每一位数
# 下面用代码来实现 n个元素的第k个排列
# 2 - 1 = ？
# 1 % 2! = 0 ~ 1 1
# 1 % 1! = 1 ~ 0 3

def permutation_seq_2(n=0, k=0):
    if k == 0 or n == 0:
        return '没有这样的排列'
    A = list(range(n + 1))
    for i in range(n):
        A[i] = i
    result = k - 1
    i = n
    s = ''
    while n > 1:
        a = result // factorial(n - 1) + 1  # 如果 = 1 说明
        result = result % factorial(n - 1)
        n -= 1
        s = s + ' ' + A[a].__str__()
        del A[a]  # 删除数组中指定位置的元素
    s = s + ' ' + A[1].__str__()
    print(s)
    k -= 1
    if k > 0:
        permutation_seq_2(i, k)


print(permutation_seq_2(3, 3))
