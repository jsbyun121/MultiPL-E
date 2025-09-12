QUERY_JL = 'concatenating two strings'
DOC_JL = """
## [Concatenation](#man-concatenation)
One of the most common and useful string operations is concatenation:
```julia-repl
julia> greet = "Hello"
"Hello"

julia> whom = "world"
"world"

julia> string(greet, ", ", whom, ".\n")
"Hello, world.\n"
```
It's important to be aware of potentially dangerous situations such as concatenation of invalid UTF-8 strings. The resulting string may contain different characters than the input strings, and its number of characters may be lower than sum of numbers of characters of the concatenated strings, e.g.:
```julia-repl
julia> a, b = "\xe2\x88", "\x80"
("\xe2\x88", "\x80")

julia> c = string(a, b)
"∀"

julia> collect.([a, b, c])
3-element Vector{Vector{Char}}:
 ['\xe2\x88']
 ['\x80']
 ['∀']

## [Concatenation](#man-concatenation)
In mathematics, `+` usually denotes a *commutative* operation, where the order of the operands does not matter. An example of this is matrix addition, where `A + B == B + A` for any matrices `A` and `B` that have the same shape. In contrast, `*` typically denotes a *noncommutative* operation, where the order of the operands *does* matter. An example of this is matrix multiplication, where in general `A * B != B * A`. As with matrix multiplication, string concatenation is noncommutative: `greet * whom != whom * greet`. As such, `*` is a more natural choice for an infix string concatenation operator, consistent with common mathematical use.

### [Concatenation](#man-array-concatenation)
| Syntax                 | Function                                   | Description                                                                                                |
|:-----------------------|:-------------------------------------------|:-----------------------------------------------------------------------------------------------------------|
|                        | [`cat`](../../base/arrays/#Base.cat)       | concatenate input arrays along dimension(s) `k`                                                            |
| `[A; B; C; ...]`       | [`vcat`](../../base/arrays/#Base.vcat)     | shorthand for `cat(A...; dims=1)`                                                                          |
| `[A B C ...]`          | [`hcat`](../../base/arrays/#Base.hcat)     | shorthand for `cat(A...; dims=2)`                                                                          |
"""


QUERY_R = "concatenating two strings"
DOC_R = """
##### [(9)](#DOCF9)
`paste(..., collapse=ss)` joins the arguments into a single character string putting `ss` in between, e.g., `ss <- "|"`. There are more tools for character manipulation, see the help for `sub` and `substring`.

Character vectors may be concatenated into a vector by the `c()` function; examples of their use will emerge frequently.
The `paste()` function takes an arbitrary number of arguments and concatenates them one by one into character strings. Any numbers given among the arguments are coerced into character strings in the evident way, that is, in the same way they would be if they were printed. The arguments are by default separated in the result by a single blank character, but this can be changed by the named argument, `sep=string`, which changes it to `string`, possibly empty.
For example
```r
> labs <- paste(c("X","Y"), 1:10, sep="")
```
makes `labs` into the character vector
```r
c("X1", "Y2", "X3", "Y4", "X5", "Y6", "X7", "Y8", "X9", "Y10")
```
Note particularly that recycling of short lists takes place here too; thus `c("X", "Y")` is repeated 5 times to match the sequence `1:10`. [9](#FOOT9)

#### 6.2.1 Concatenating lists
When the concatenation function `c()` is given list arguments, the result is an object of mode list also, whose components are those of the argument lists joined together in sequence.
```r
> list.ABC <- c(list.A, list.B, list.C)
```
Recall that with vector objects as arguments the concatenation function similarly joined together all arguments into a single vector structure. In this case all other attributes, such as `dim` attributes, are discarded.
"""


QUERY_RKT = "concatenating two strings"
DOC_RKT = """
strs : (listof string?)
  sep : string? = " "
  before-first : string? = ""
  before-last : string? = sep
  after-last : string? = ""```


Appends the strings in strs, inserting sep between each pair of strings in strs. before-last, before-first, and after-last are analogous to the inputs of [add-between](pairs.html#%28def._%28%28lib._racket%2Flist..rkt%29._add-between%29%29): they specify an alternate separator between the last two strings, a prefix string, and a suffix string respectively.

Examples:

> ```racket
> > ( string-join ' ( "one" "two" "three" "four" ) )
> "one two three four"
> > ( string-join ' ( "one" "two" "three" "four" ) ", " )
> "one, two, three, four"
> > ( string-join ' ( "one" "two" "three" "four" ) " potato " )
> "one potato two potato three potato four"
> > ( string-join ' ( "x" "y" "z" ) ", " #:before-first "Todo: " #:before-last " and " #:after-last "." )
> "Todo: x, y and z."
> ```

# 4.5 Byte Strings
Appends the byte strings in strs, inserting sep between each pair of bytes in strs. A new mutable byte string is returned.
Example:
> ```racket
> > ( bytes-join ' ( #"one" #"two" #"three" #"four" ) #" potato " )
> #"one potato two potato three potato four"
> ```
------------------------------------------------------------------------

The bindings documented in this section are provided by the [racket/string](#%28mod-path._racket%2Fstring%29) and [racket](index.html) libraries, but not [racket/base](index.html).
> ```
(string-append* str ... strs) → string?
  str : string?
  strs : (listof string?)
"""


QUERY_ML = "concatenating two strings"
DOC_ML = """
### 2.1 Strings
Another difficulty is the implementation of the method concat. In order to concatenate a string with another string of the same class, one must be able to access the instance variable externally. Thus, a method repr returning s must be defined. Here is the correct definition of strings:
```ocaml
# class ostring s = object (self : 'mytype) val repr = s method repr = repr method get n = String.get repr n method print = print_string repr method escaped = {\< repr = String.escaped repr \>} method sub start len = {\< repr = String.sub s start len \>} method concat (t : 'mytype) = {\< repr = repr ^ t#repr \>} end;;
```
class ostring : string -> object ('a) val repr : string method concat : 'a -> 'a method escaped : 'a method get : int -> char method print : unit method repr : string method sub : int -> int -> 'a end
Another constructor of the class string can be defined to return a new string of a given length:
```ocaml
# class cstring n = ostring (String.make n ' ');;
"""


QUERY_LUA = "concatenating two strings"
DOC_LUA = """
### `table.concat (list [, sep [, i [, j]]])`
Given a list where all elements are strings or numbers, returns the string `list[i]..sep..list[i+1] ··· sep..list[j]`. The default value for `sep` is the empty string, the default for `i` is 1, and the default for `j` is `#list`. If `i` is greater than `j`, returns the empty string.
------------------------------------------------------------------------

### 3.4.6 – Concatenation
The string concatenation operator in Lua is denoted by two dots ('`..`'). If both operands are strings or numbers, then the numbers are converted to strings in a non-specified format (see [§3.4.3](#3.4.3)). Otherwise, the `__concat` metamethod is called (see [§2.4](#2.4)).

### `string.rep (s, n [, sep])`
Returns a string that is the concatenation of `n` copies of the string `s` separated by the string `sep`. The default value for `sep` is the empty string (that is, no separator). Returns the empty string if `n` is not positive.
(Note that it is very easy to exhaust the memory of your machine with a single call to this function.)
"""