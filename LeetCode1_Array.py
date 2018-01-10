# Given a sorted array, remove the duplicates in place such that each element appear only once
# and return the new length.
# Do not allocate extra space for another array, you must do this in place with constant memory.
# For example, Given input array A = [1,1,2],
# Your function should return length = 2, and A is now [1,2].
from re import L

import sys
from numpy import sort


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
    B = sort(A)
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
    B = sort(A)
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
            print('resulet %s and value %s' % (result,value))
            if result < value:
                ss = '当前记录的大小 %s  %s %s ' % (x, B[j], B[k])
                value = result
            if x + B[j] + B[k] < target:
                j += 1
            else:
                k -= 1

    print(ss)

print(three_sum_closest([-1, 2, 1, -4], 1))
