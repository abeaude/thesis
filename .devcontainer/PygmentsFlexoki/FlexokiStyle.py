from pygments.style import Style
from pygments.token import Token, Comment, Keyword, Name, String, Error, Generic, Number, Operator, Punctuation
# https://pygments.org/docs/tokens/
class FlexokiStyle(Style):
    name = "Flexoki"
    background_color = "#FFFCF0"
    styles = {
        Comment: "#B7B5AC",
        Punctuation: "#6F6E69",
        
        Operator: '#AF3029',
        Operator.Word: "bold #AF3029",
        
        # Name: "",
        Name.Function: "bold #BC5215",
        Name.Function.Magic: "",
        Name.Builtin: "#24837B",
        Name.Builtin.Pseudo: "",
        Name.Decorator: "bold #AD8301",
        Name.Class: "#BC5215",
        Name.Constant: "#AD8301",
        Name.Exception: "#AF3029",
        # Name.Namespace: "",
        
        
        Keyword: "#66800B",
        Keyword.Namespace: "bold #AF3029",
        Keyword.Constant: "#AD8301",
        Keyword.Reserved: "#A02F6F",
        # Keyword.Type: "",
        
        Number: "#5E409D",
        String: "#24837B",
        # String.Doc: "",
        # String.Double: "",
        # String.Single: "",
        # String.Escape: "",
        # String.Interpol: "",

    }