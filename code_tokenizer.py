import code_tokenize as ctok

# Python
py = ctok.tokenize(
    '''
        def my_func():
            print("Hello World")
    ''',
    lang = "python")
print(py)

# Output: [def, my_func, (, ), :, #NEWLINE#, ...]
        # [def, my_func, (, ), :, #INDENT#, print, (, "Hello World", ), #NEWLINE#, #DEDENT#]

# Java
jav = ctok.tokenize(
    '''
        public static void main(String[] args){
          System.out.println("Hello World");
        }
    ''',
    lang = "java", 
    syntax_error = "ignore")

print(jav)

# Output: [public, static, void, main, (, String, [, ], args), {, System, ...]

# JavaScript
js = ctok.tokenize(
    '''
        alert("Hello World");
    ''',
    lang = "javascript", 
    syntax_error = "ignore")
print(js)

# Output: [alert, (, "Hello World", ), ;]

