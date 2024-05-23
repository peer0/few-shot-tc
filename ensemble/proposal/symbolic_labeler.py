import re

class TimeComplexityCalculator:
    def __init__(self, code):
        self.code = code

    def extract_functions(self):
        functions = []
        lines = self.code.split('\n')
        current_function = ''
        for line in lines:
            if line.strip().startswith('def '):
                if current_function:
                    functions.append(current_function.strip())
                current_function = line
            else:
                current_function += '\n' + line
        if current_function:
            functions.append(current_function.strip())
        return functions

    def extract_loop_counts_and_function_calls(self, func_code):
        loop_counts = []
        function_calls = []
        lines = func_code.split('\n')
        for line in lines:
            if 'for' in line and 'range' in line:
                match = re.search(r'range\s*\(\s*(.*?)\s*\)', line)
                if match:
                    loop_counts.append(match.group(1).split(','))
            elif 'while' in line:
                match = re.search(r'while\s+(.*):', line)
                if match:
                    condition = match.group(1)
                    loop_counts.append([condition])
            func_call_match = re.search(r'(\w+)\s*\(', line)
            if func_call_match and not line.strip().startswith('def '):
                function_calls.append(func_call_match.group(1))
        return loop_counts, function_calls

    def extract_loop_counts(self):
        loop_counts = []
        lines = self.code.split('\n')
        for line in lines:
            if 'for' in line and 'range' in line:
                match = re.search(r'range\s*\(\s*(.*?)\s*\)', line)
                if match:
                    loop_counts.append(match.group(1).split(','))
            elif 'while' in line:
                match = re.search(r'while\s+(.*):', line)
                if match:
                    condition = match.group(1)
                    loop_counts.append([condition])
        return loop_counts

    def calculate_time_complexity_with_function_calls(self):
        functions = self.extract_functions()
        function_call_counts = {func: 0 for func in functions}
        function_loop_counts = {}
        function_call_names = {}

        for func_code in functions:
            match = re.search(r'def (\w+)\(', func_code)
            if not match:
                continue
            function_name = match.group(1)
            loop_counts, function_calls = self.extract_loop_counts_and_function_calls(func_code)
            function_loop_counts[function_name] = loop_counts
            function_call_names[function_name] = function_calls

        for func_name, calls in function_call_names.items():
            for call in calls:
                if call in function_call_counts:
                    function_call_counts[call] += 1  # 호출 횟수 갱신

        time_complexity_parts = []
        for func_name, loop_counts in function_loop_counts.items():
            time_complexity_part = []
            for count in loop_counts:
                time_complexity_part.append(" * ".join(count))
            for call_name in function_call_names.get(func_name, []):
                if call_name in function_call_counts:
                    time_complexity_part.append(f"{function_call_counts[call_name]}")
            time_complexity_parts.append("(" + " * ".join(time_complexity_part) + ")")

        if time_complexity_parts:
            total_complexity = " + ".join(time_complexity_parts)

        return total_complexity

    def calculate_time_complexity(self):
        loop_counts = self.extract_loop_counts()
        time_complexity_parts = []

        for counts in loop_counts:
            time_part = ""
            for count in counts:
                time_part += count + " * "
            time_complexity_parts.append(time_part[:-3])  # Remove the last " * "

        if time_complexity_parts:
            total_complexity = " * ".join(time_complexity_parts)

        return total_complexity

    def classify_time_complexity(self, time_complexity):
        parts = time_complexity.split(" * ")

        # Count the occurrences of numbers and letters
        num_count = 0
        char_count = 0
        for part in parts:
            if part.isdigit():
                num_count += 1
            elif re.match(r'log\(\d+\)', part):
                return "logn"
            elif re.match(r'\d+ \* log\(\d+\)', part):
                return "nlogn"
            elif part.isalpha():
                char_count += 1

        # Classify the time complexity
        if num_count >= 1 and char_count == 0:
            return "constant"
        elif num_count >= 1 and char_count == 1:
            return "linear"
        elif num_count >= 1 and char_count == 2:
            return "quadratic"
        elif num_count >= 1 and char_count == 3:
            return "cubic"
        else:
            return "np"

# 입력이 함수를 포함하는지 아닌지, 재귀호출 여부 고려
def process_code(code):
    calculator = TimeComplexityCalculator(code)
    # if "def " in code:
    #     time_complexity = calculator.calculate_time_complexity_with_function_calls()
    # else:
    #     time_complexity = calculator.calculate_time_complexity()
    time_complexity = calculator.calculate_time_complexity()
    return calculator.classify_time_complexity(time_complexity)



# unit test
if __name__ == "__main__":

    code_sample_with_functions = """
    import sys
    import math

    prime=[True for _ in range(1000001)]

    # # Remove these 2 lines while submitting your code online
    # sys.stdin = open('input.txt', 'r')
    # sys.stdout = open('output.txt', 'w')
    def solve():
        n,e,h,a,b,c=map(int,input().split())
        ans=1e9
        for i in range(1,1000001):
            su=0
            ntmp=n
            tmp1=e
            tmp2=h
            tmp1-=i
            tmp2-=i
            if (tmp1<0 or tmp2<0 or i>ntmp):
                break
            ntmp-=i
            su+=(c*i)
            if (ntmp==0):
                ans=min(ans,su)
                continue
            if (a<=b):
                if ((tmp1//2)>=ntmp):
                    su+=int(a*ntmp)
                    ntmp-=ntmp
                else:
                    su+=int(a*(tmp1//2))
                    ntmp-=(tmp1//2)
                    if (ntmp<=(tmp2//3)):
                        su+=int(b*ntmp)
                        ntmp-=ntmp
                    else:
                        su+=int(b*(tmp2//3))
                        ntmp-=(tmp2//3)
            else:
                if ((tmp2//3)>=ntmp):
                    su+=int(b*ntmp)
                    ntmp-=ntmp
                else:
                    su+=int(b*(tmp2//3))
                    ntmp-=(tmp2//3)
                    if (ntmp<=(tmp1//2)):
                        su+=int(a*ntmp)
                        ntmp-=ntmp
                    else:
                        su+=int(a*(tmp1//2))
                        ntmp-=(tmp1//2)
            if (ntmp==0):
                ans=min(ans,su)
        # print(ans)
        if (ans==1e9):
            print("-1")
        else:
            print(ans)

    def main():
        n=int(input())
        s=input()
        m={}
        have={}
        cc=0
        for c in s:
            if (c not in m):
                m[c]=1
            else:
                m[c]+=1
        ct=len(m)
        l=0
        ans=1e9
        for i in range(0,n):
            solve()
            if (s[i] not in have):
                have[s[i]]=0
                cc+=1
            have[s[i]]+=1
            while(l<=i and have[s[l]]>1):
                have[s[l]]-=1
                l+=1
            if (cc==ct):
                ans=min(ans,i-l+1)

        print(ans)

    if __name__ == "__main__":
        main()
    """

    code_sample_without_functions = """
    from sys import stdin, stdout
    from math import sin, tan, cos

    n, m, k, l = map(int, stdin.readline().split())

    lb, rb = 0, n // m + 1
    while rb - lb > 1:
        mid = (lb + rb) >> 1
        
        if mid * m - k >= l:
            rb = mid
        else:
            lb = mid

    if lb != n // m:
        stdout.write(str(rb))
    else:
        stdout.write('-1')
    """

    # 함수를 포함하는 경우
    result_with_functions = process_code(code_sample_with_functions)
    print("With functions:", result_with_functions)

    # 함수를 포함하지 않는 경우
    result_without_functions = process_code(code_sample_without_functions)
    print("Without functions:", result_without_functions)

