---
title: "Grind 75 Questions"
date: 2023-0911:29:21+08:00
draft: false
ShowToc: true
category: [coding, interview]
tags: ["coding", "grind75", "interview"]
description: "Grind 75 Questions for Interview Prep"
summary: "Grind 75 Questions for Interview Prep"
---


# Week1

## 1. [Two Sum](https://leetcode.com/problems/two-sum)

- Array

### Problem

Given an array of integers `nums` and an integer `target`, return *indices of the two numbers such that they add up to `target`*.

You may assume that each input would have ***exactly* one solution**, and you may not use the *same* element twice.

You can return the answer in any order.

**Example 1:**

```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

```

**Example 2:**

```
Input: nums = [3,2,4], target = 6
Output: [1,2]

```

**Example 3:**

```
Input: nums = [3,3], target = 6
Output: [0,1]

```

**Constraints:**

- `2 <= nums.length <= 104`
- `109 <= nums[i] <= 109`
- `109 <= target <= 109`
- **Only one valid answer exists.**

**Follow-up:** Can you come up with an algorithm that is less than `O(n2)` time complexity?

### Solution

```python

from collections import defaultdict

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # # Let's try a simple solution first
        # # O(n^2) - Time, O(1) - space
        # for i, first in enumerate(nums):
        #     for j, second in enumerate(nums):
        #         if i == j: continue
        #         if first + second == target:
        #             return [i, j]

        ## Two pass hash map
        ## O(n^2) - Time, O(n) space
        # data = defaultdict(list)
        # for i, x in enumerate(nums):
        #     data[x].append(i)
        # for k, v in data.items():
        #     chk_s = target - k
        #     diff = data.get(chk_s)
        #     if diff:
        #         if len(diff) > 1:
        #             return [diff[0], diff[1]]
        #         else:
        #             if chk_s != k:
        #                 return [v[0], data[chk_s][0]]

        ## Cleaner hashmap solution
        ## Two pass hashmap
        ## O(n) - time , O(n) - space

        # data = {}
        # for i, x in enumerate(nums):
        #     data[x] = i

        # for i, x in enumerate(nums):
        #     diff = target - x
        #     if diff in data and i != data[diff]:
        #         return [i, data[diff]]

        ## Cleaner hashmap solution
        ## One pass hashmap
        ## O(n) - time , O(n) - space

        data = {}
        for i, x in enumerate(nums):
            diff = target - x
            if diff in data:
                return [i, data[diff]]
            data[x] = i
```

## 2. [Valid Parentheses](https://leetcode.com/problems/valid-parentheses)

### Problem

- Stack

Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

An input string is valid if:

1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

**Example 1:**

```
Input: s = "()"
Output: true

```

**Example 2:**

```
Input: s = "()[]{}"
Output: true

```

**Example 3:**

```
Input: s = "(]"
Output: false

```

**Constraints:**

- `1 <= s.length <= 104`
- `s` consists of parentheses only `'()[]{}'`.

### Solution

```python
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        stack = list()
        for brac in s:
            if stack:
                if brac == ")" and stack[-1] == "(":
                    stack.pop()
                elif brac == "}" and stack[-1] == "{":
                    stack.pop()
                elif brac == "]" and stack[-1] == "[":
                    stack.pop()
                else:
                    stack.append(brac)
            else:
                stack.append(brac)

        if stack:
            return False
        else:
            return True
```

## 3. [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists)

- Linked List

### Problem

You are given the heads of two sorted linked lists `list1` and `list2`.

Merge the two lists into one **sorted** list. The list should be made by splicing together the nodes of the first two lists.

Return *the head of the merged linked list*.

```
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

```

**Example 2:**

```
Input: list1 = [], list2 = []
Output: []

```

**Example 3:**

```
Input: list1 = [], list2 = [0]
Output: [0]

```

**Constraints:**

- The number of nodes in both lists is in the range `[0, 50]`.
- `100 <= Node.val <= 100`
- Both `list1` and `list2` are sorted in **non-decreasing** order.

### Solution

```python
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        idx1 = 0
        idx2 = 0
        merged_ll = None
        start = None
        while True:

            if list1 == None and list2 == None:
                break
            
            if merged_ll is None:
                start = merged_ll = ListNode()
            else:
                merged_ll.next = ListNode()
                merged_ll = merged_ll.next

            if list1 != None and list2 != None:
                if list1.val < list2.val:
                    merged_ll.val = list1.val
                    list1 = list1.next
                else:
                    merged_ll.val = list2.val
                    list2 = list2.next
            elif list1 == None:
                merged_ll.val = list2.val
                list2 = list2.next
            else:
                merged_ll.val = list1.val
                list1 = list1.next

        return start
```

## 4. [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock)

- Array

### Problem

You are given an array `prices` where `prices[i]` is the price of a given stock on the `ith` day.

You want to maximize your profit by choosing a **single day** to buy one stock and choosing a **different day in the future** to sell that stock.

Return *the maximum profit you can achieve from this transaction*. If you cannot achieve any profit, return `0`.

**Example 1:**

```
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

```

**Example 2:**

```
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

```

**Constraints:**

- `1 <= prices.length <= 10^5`
- `0 <= prices[i] <= 10^4`

### Solution

```python
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        ## O(n^2) don't pass
        # max_profit = 0
        # for i, x in enumerate(prices):
        #     for y in prices[i:]:
        #         max_profit = max(max_profit, y-x)

        # return int(max_profit)

        # O(n) - time , O(1) - space
        price = prices[0]
        max_profit = 0

        for x in prices:
            if x < price:
                price = x
                continue
            potential_profit = x - price
            if potential_profit > max_profit:
                max_profit = potential_profit
            
        return max_profit
```

## 5. [Valid Palindrome](https://leetcode.com/problems/valid-palindrome)

### Problem

A phrase is a **palindrome** if, after converting all uppercase letters into lowercase letters and 
removing all non-alphanumeric characters, it reads the same forward and 
backward. Alphanumeric characters include letters and numbers.

Given a string `s`, return `true` *if it is a **palindrome**, or* `false` *otherwise*.

**Example 1:**

```
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

```

**Example 2:**

```
Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

```

**Example 3:**

```
Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.

```

**Constraints:**

- `1 <= s.length <= 2 * 10^5`
- `s` consists only of printable ASCII characters.

### Solution

```python
class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # Fast and accepted
        # s = s.lower()
        # s = [x for x in s if x.isalnum()]
        # # print(s)
        # len_s = len(s)
        # for i in range(len_s//2):
        #     if s[i] == s[-i-1]:
        #         continue
        #     else:
        #         return False

        # return True

        # Faster
        s = s.lower()
        s = [x for x in s if x.isalnum()]
        len_s = len(s)
        return all (s[i] == s[-i-1] for i in range(len_s//2))
```

## 6. [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree)

## 7. [Valid Anagram](https://leetcode.com/problems/valid-anagram)

## 8. [Binary Search](https://leetcode.com/problems/binary-search)

## 9. [Flood Fill](https://leetcode.com/problems/flood-fill)

## 10. [Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree)

## 11. [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree)

## 12. [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle)

## 13. [Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks)

# References

1. [Grind 75 - A better Blind 75 you can customize, by the author of Blind 75](https://www.techinterviewhandbook.org/grind75)