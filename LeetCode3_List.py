# LeetCode里面的链表相关的算法题


# Add Two Numbers
# You are given two linked lists representing two non-negative numbers. The digits are stored in reverse
# order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
# Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
# Output: 7 -> 0 -> 8
# 这个题主要是类似数据在机器中存储的方式，比如常见的342，在链表中是逆向存储的，所以就成了 2—>4—>3这样
# 同样 5—>6—>4 就是465，我们就会发现342 + 465 = 807，在十位上相加是超过10向前进一，但是链表是逆向的，所以就是向后进一
#
# 其实可以将两个链表转成整型 相加 然后将结果倒序 插入新的链表中即可
#
# 链表的节点
class ListNode(object):

    def __init__(self, x):
        self.val = x
        self.next = None


def two_number_add(a, b):
    numLine1 = []
    numLine2 = []
    while a is not None:
        numLine1.append(a.val)
        a = a.next
    while b is not None:
        numLine2.append(b.val)
        b = b.next

    num1 = 0
    num2 = 0
    for n in range(len(numLine1)):
        num1 += numLine1[n] * (10 ** n)  # 10 ** n表示 10 的n次方

    for m in range(len(numLine2)):
        num2 += numLine2[m] * (10 ** m)

    num3 = num1 + num2
    # 构造链表返回
    L3 = ListNode
    first = ListNode(0)
    L3.val = 'start'
    L3.next = first
    while num3 >= 10:
        temp = ListNode(None)
        first.val = num3 % 10
        first.next = temp
        first = temp
        num3 = num3 // 10

    first.val = num3 % 10  # 此时的num3 < 10
    return L3


a = ListNode(0)
b = ListNode(0)
c = ListNode(0)
a.val = 2
a.next = b

b.val = 4
b.next = c

c.val = 3
c.next = None

l = ListNode(0)
m = ListNode(0)
n = ListNode(0)
l.val = 5
l.next = b
m.val = 6
m.next = c
n.val = 4
n.next = None

z = two_number_add(a, l)

while z is not None:
    print('%s  ' % z.val)
    z = z.next