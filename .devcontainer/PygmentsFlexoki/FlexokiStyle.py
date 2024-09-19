from pygments.style import Style
from pygments.token import Token, Comment, Keyword, Name, String, Error, Generic, Number, Operator, Punctuation

class FlexokiStyle(Style):
    name = "Flexoki"
    background_color = "#FFFCF0"
    styles = {
        Comment: "#B7B5AC",
        Punctuation: "#6F6E69",
        
        Operator: 'bold #6F6E69',
        
        Name.Function: "bold #BC5215",
        Name.Builtin: "#24837B",
        Name.Decorator: "bold #AD8301",
        
        Keyword: "#205EA6",
        Keyword.Namespace: "#AF3029",
        
        Number: "#5E409D",
        String: "#24837B",
    }