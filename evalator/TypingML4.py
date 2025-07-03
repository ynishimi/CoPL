# typingml4.py
import sys
import re
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional
import copy

# -------------------------------------------------------------------
# 1. 型の構造 (Types)
# -------------------------------------------------------------------
class Type:
    def __str__(self): return "unknown"
    def __eq__(self, other):
        # 型の等価性比較は、型の文字列表現に依存します。
        # これにより、解決済みの型変数も正しく比較されます。
        if not isinstance(other, Type): return False
        return str(self) == str(other)
    def __hash__(self):
        return hash(str(self))

@dataclass(frozen=True)
class TInt(Type):
    def __str__(self): return "int"

@dataclass(frozen=True)
class TBool(Type):
    def __str__(self): return "bool"

@dataclass(frozen=True)
class TArrow(Type):
    param_type: Type
    return_type: Type
    def __str__(self):
        # 引数型が関数型の場合、括弧で囲みます。
        # [修正点] resolve を使って型変数の先をたどり、正しく関数型か判定します。
        p_type_resolved = resolve(self.param_type)
        p_str = str(self.param_type)
        if isinstance(p_type_resolved, TArrow):
            p_str = f"({p_str})"
        return f"{p_str} -> {self.return_type}"

@dataclass(frozen=True)
class TList(Type):
    element_type: Type
    def __str__(self):
        # 要素型が関数型の場合、括弧で囲みます。
        # [修正点] resolve を使って型変数の先をたどり、正しく関数型か判定します。
        e_type_resolved = resolve(self.element_type)
        e_str = str(self.element_type)
        if isinstance(e_type_resolved, TArrow):
             e_str = f"({e_str})"
        return f"{e_str} list"

@dataclass(frozen=True)
class TError(Type):
    message: str = "error"
    def __str__(self): return self.message

class TVar(Type):
    """型推論のための一時的な型変数を表すクラス。"""
    _id_counter = 0
    def __init__(self):
        self.id = TVar._id_counter
        TVar._id_counter += 1
        self.instance: Optional[Type] = None
    
    def __str__(self):
        # 型変数が解決済み（具体型に束縛済み）であれば、その具体型を返します。
        # 未解決であれば、't0, 't1 のような名前を返します。
        if self.instance:
            return str(resolve(self.instance))
        return f"'t{self.id}"

# -------------------------------------------------------------------
# 2. 式の構造 (AST)
# -------------------------------------------------------------------
class Expression:
    def to_string(self) -> str: return str(self)
    def get_precedence(self): return 99

# ... (ASTクラスは変更なし) ...
@dataclass
class IntLiteral(Expression):
    value: int
    def __str__(self): return str(self.value)

@dataclass
class BoolLiteral(Expression):
    value: bool
    def __str__(self): return str(self.value).lower()

@dataclass
class Variable(Expression):
    name: str
    def __str__(self): return self.name

precedence = {'App': 5, 'Times': 4, 'Plus': 3, 'Minus': 3, 'Cons': 2, 'LessThan': 1, 'If': 0, 'Let': 0, 'LetRec': 0, 'Fun': 0, 'Match': 0}

@dataclass
class BinaryOp(Expression):
    op: str; left: Expression; right: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self) -> str:
        my_prec = self.get_precedence()
        left_str, right_str = self.left.to_string(), self.right.to_string()
        if self.left.get_precedence() < my_prec: left_str = f"({left_str})"
        if self.op == '::':
            if self.right.get_precedence() < my_prec: right_str = f"({right_str})"
        else:
            if self.right.get_precedence() <= my_prec: right_str = f"({right_str})"
        return f"{left_str} {self.op} {right_str}"

class Plus(BinaryOp):
    def __init__(self, left, right): super().__init__('+', left, right)
class Minus(BinaryOp):
    def __init__(self, left, right): super().__init__('-', left, right)
class Times(BinaryOp):
    def __init__(self, left, right): super().__init__('*', left, right)
class LessThan(BinaryOp):
    def __init__(self, left, right): super().__init__('<', left, right)
class Cons(BinaryOp):
    def __init__(self, left, right): super().__init__('::', left, right)

@dataclass
class Nil(Expression):
    def __str__(self): return "[]"
    def to_string(self): return "[]"

@dataclass
class If(Expression):
    cond: Expression; true_branch: Expression; false_branch: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self): return f"if {self.cond.to_string()} then {self.true_branch.to_string()} else {self.false_branch.to_string()}"

@dataclass
class Let(Expression):
    var: str; bound_expr: Expression; body_expr: Expression; var_type: Optional[Type] = None
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self):
        if self.var_type:
            return f"let {self.var} : {self.var_type} = {self.bound_expr.to_string()} in {self.body_expr.to_string()}"
        return f"let {self.var} = {self.bound_expr.to_string()} in {self.body_expr.to_string()}"

@dataclass
class Fun(Expression):
    arg_name: str; body: Expression; arg_type: Optional[Type] = None
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self):
        # ASTのto_stringでは型変数を表示したくないので、解決済みの型を使う
        arg_type_str = ""
        if self.arg_type:
            arg_type_str = f" : {resolve(self.arg_type)}"
        return f"fun {self.arg_name}{arg_type_str} -> {self.body.to_string()}"


@dataclass
class App(Expression):
    func_expr: Expression; arg_expr: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self) -> str:
        left_str, right_str = self.func_expr.to_string(), self.arg_expr.to_string()
        if self.func_expr.get_precedence() < self.get_precedence(): left_str = f"({left_str})"
        if self.arg_expr.get_precedence() <= self.get_precedence(): right_str = f"({right_str})"
        return f"{left_str} {right_str}"

@dataclass
class LetRec(Expression):
    func_name: str; func_type: Type; arg_name: str; arg_type: Type; func_body: Expression; let_body: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self): return f"let rec {self.func_name} : {self.func_type} = fun ({self.arg_name} : {self.arg_type}) -> {self.func_body.to_string()} in {self.let_body.to_string()}"

@dataclass
class Match(Expression):
    match_expr: Expression; nil_branch: Expression; head_var: str; tail_var: str; cons_branch: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self): return f"match {self.match_expr.to_string()} with [] -> {self.nil_branch.to_string()} | {self.head_var} :: {self.tail_var} -> {self.cons_branch.to_string()}"

TypeEnv = List[Tuple[str, Type]]
Token = Tuple[str, str]

# -------------------------------------------------------------------
# 3. 導出規則 (Derivation)
# -------------------------------------------------------------------
def format_tenv(tenv: TypeEnv) -> str:
    """
    型環境を整形して文字列にします。
    [修正点] 表示する前に各型を resolve し、推論結果を反映させます。
    """
    if not tenv: return ""
    return ", ".join(f"{var}:{resolve(t)}" for var, t in tenv) + " "

class Derivation:
    def format(self, i=0) -> str: raise NotImplementedError

@dataclass
class TGeneric(Derivation):
    tenv: TypeEnv; expr: Expression; type: Type; rule_name: str; premises: List[Derivation]
    def format(self, i=0) -> str:
        indent = " " * i
        # 表示する際は、常に解決済みの型を使う
        resolved_type = resolve(self.type)
        if not self.premises:
            return f"{indent}{format_tenv(self.tenv)}|- {self.expr.to_string()} : {resolved_type} by {self.rule_name} {{}};"
        else:
            premise_str = "\n".join(p.format(i + 1) for p in self.premises)
            return f"{indent}{format_tenv(self.tenv)}|- {self.expr.to_string()} : {resolved_type} by {self.rule_name} {{\n{premise_str}\n{indent}}};"

# -------------------------------------------------------------------
# 4. 型推論エンジン (Unification-based Type Inference)
# -------------------------------------------------------------------

def resolve(t: Type) -> Type:
    """型変数の連鎖をたどり、最終的な型を返す。"""
    if isinstance(t, TVar) and t.instance:
        t.instance = resolve(t.instance)
        return t.instance
    return t

def unify(t1: Type, t2: Type):
    """二つの型を単一化する。"""
    t1 = resolve(t1)
    t2 = resolve(t2)

    if isinstance(t1, TVar):
        if t1 != t2: t1.instance = t2
        return
    if isinstance(t2, TVar):
        t2.instance = t1
        return
    if isinstance(t1, TArrow) and isinstance(t2, TArrow):
        unify(t1.param_type, t2.param_type)
        unify(t1.return_type, t2.return_type)
        return
    if isinstance(t1, TList) and isinstance(t2, TList):
        # [] の型推論のために、片方が未解決のリスト型の場合を考慮
        if isinstance(resolve(t1.element_type), TError):
             unify(t1.element_type, t2.element_type)
        elif isinstance(resolve(t2.element_type), TError):
             unify(t2.element_type, t1.element_type)
        else:
             unify(t1.element_type, t2.element_type)
        return
    if t1 != t2:
        raise TypeError(f"型が一致しません: {t1} と {t2}")


def type_infer(tenv: TypeEnv, expr: Expression, hint_type: Optional[Type] = None) -> Tuple[Type, Derivation]:
    if isinstance(expr, IntLiteral): return TInt(), TGeneric(tenv, expr, TInt(), "T-Int", [])
    if isinstance(expr, BoolLiteral): return TBool(), TGeneric(tenv, expr, TBool(), "T-Bool", [])
    if isinstance(expr, Nil):
        elem_type = TVar()
        list_type = TList(elem_type)
        if hint_type: unify(list_type, hint_type)
        return list_type, TGeneric(tenv, expr, list_type, "T-Nil", [])

    if isinstance(expr, Variable):
        for var, t in reversed(tenv):
            if var == expr.name: return t, TGeneric(tenv, expr, t, "T-Var", [])
        raise TypeError(f"未定義の変数です: {expr.name}")

    if isinstance(expr, BinaryOp):
        t1, d1 = type_infer(tenv, expr.left)
        t2, d2 = type_infer(tenv, expr.right)
        if expr.op in ['+', '-', '*']:
            unify(t1, TInt())
            unify(t2, TInt())
            rule_map = {'+': 'T-Plus', '-': 'T-Minus', '*': 'T-Times'}
            return TInt(), TGeneric(tenv, expr, TInt(), rule_map[expr.op], [d1, d2])
        if expr.op == '<':
            unify(t1, TInt())
            unify(t2, TInt())
            return TBool(), TGeneric(tenv, expr, TBool(), "T-Lt", [d1, d2])
        if expr.op == '::':
            list_type = TList(t1)
            unify(list_type, t2)
            return resolve(t2), TGeneric(tenv, expr, resolve(t2), "T-Cons", [d1, d2])

    if isinstance(expr, If):
        t_cond, d_cond = type_infer(tenv, expr.cond)
        unify(t_cond, TBool())
        t_true, d_true = type_infer(tenv, expr.true_branch, hint_type=hint_type)
        t_false, d_false = type_infer(tenv, expr.false_branch, hint_type=hint_type)
        unify(t_true, t_false)
        res_type = resolve(t_true)
        return res_type, TGeneric(tenv, expr, res_type, "T-If", [d_cond, d_true, d_false])

    if isinstance(expr, Let):
        t1, d1 = type_infer(tenv, expr.bound_expr, hint_type=expr.var_type)
        if expr.var_type:
            unify(t1, expr.var_type)
        
        resolved_t1 = resolve(t1)
        t2, d2 = type_infer(tenv + [(expr.var, resolved_t1)], expr.body_expr, hint_type=hint_type)
        return t2, TGeneric(tenv, expr, t2, "T-Let", [d1, d2])

    if isinstance(expr, Fun):
        # arg_type_for_inference は、元のASTノード 'expr' を変更せずに、
        # 型推論のボディ部分で使用する型を保持します。
        # これにより、表示される出力が元のソースコードを反映するようになります。
        arg_type_for_inference = expr.arg_type
        ret_hint = None

        if isinstance(hint_type, TArrow):
            if arg_type_for_inference is None:
                arg_type_for_inference = hint_type.param_type
            else:
                unify(arg_type_for_inference, hint_type.param_type)
            ret_hint = hint_type.return_type

        if arg_type_for_inference is None:
            arg_type_for_inference = TVar()

        new_tenv = tenv + [(expr.arg_name, arg_type_for_inference)]
        ret_type, d_body = type_infer(new_tenv, expr.body, hint_type=ret_hint)
        
        func_type = TArrow(arg_type_for_inference, ret_type)
        # 導出の構築には元の 'expr' を使用するため、その文字列表現は
        # 推論プロセスによる影響を受けません。
        return func_type, TGeneric(tenv, expr, func_type, "T-Fun", [d_body])

    if isinstance(expr, App):
        t_func, d_func = type_infer(tenv, expr.func_expr)
        t_arg, d_arg = type_infer(tenv, expr.arg_expr)
        
        ret_type = TVar()
        expected_func_type = TArrow(t_arg, ret_type)
        unify(t_func, expected_func_type)
        
        final_ret_type = resolve(ret_type)
        return final_ret_type, TGeneric(tenv, expr, final_ret_type, "T-App", [d_func, d_arg])

    if isinstance(expr, LetRec):
        unify(expr.func_type, TArrow(expr.arg_type, TVar())) # func_typeが関数型であることを保証
        
        env_for_body = tenv + [(expr.func_name, expr.func_type), (expr.arg_name, expr.arg_type)]
        t_body, d_body = type_infer(env_for_body, expr.func_body)
        
        # func_typeの戻り値部分と実際の本体の戻り値を単一化
        unify(expr.func_type.return_type, t_body)
        
        env_for_in = tenv + [(expr.func_name, expr.func_type)]
        t_in, d_in = type_infer(env_for_in, expr.let_body)
        return t_in, TGeneric(tenv, expr, t_in, "T-LetRec", [d_body, d_in])

    if isinstance(expr, Match):
        t_match, d_match = type_infer(tenv, expr.match_expr)
        elem_type = TVar()
        unify(t_match, TList(elem_type))
        
        t_nil, d_nil = type_infer(tenv, expr.nil_branch)
        
        env_cons = tenv + [(expr.head_var, resolve(elem_type)), (expr.tail_var, resolve(t_match))]
        t_cons, d_cons = type_infer(env_cons, expr.cons_branch)
        
        unify(t_nil, t_cons)
        res_type = resolve(t_nil)
        return res_type, TGeneric(tenv, expr, res_type, "T-Match", [d_match, d_nil, d_cons])
        
    raise TypeError(f"型推論未対応の式の型です: {type(expr)}")

# -------------------------------------------------------------------
# 5. パーサー (Parser with Type Annotations)
# -------------------------------------------------------------------
# ... (Parserは変更なし) ...
class Parser:
    def __init__(self, tokens: List[Token]): self.tokens, self.pos = tokens, 0
    @classmethod
    def from_text(cls, text: str):
        keywords = {'if', 'then', 'else', 'true', 'false', 'let', 'in', 'fun', 'rec', 'match', 'with', 'int', 'bool', 'list'}
        token_spec = [('CONS', r'::'), ('ARROW', r'->'), ('BAR', r'\|'), ('ID', r'[a-zA-Z_][a-zA-Z0-9_]*'), ('INT', r'-?\d+'), ('OP', r'[+*<()=\[\]_:-]'), ('SKIP', r'\s+'), ('MISMATCH', r'.')]
        tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_spec)
        tokens: List[Token] = []
        for mo in re.finditer(tok_regex, text):
            kind, value = mo.lastgroup, mo.group()
            if kind == 'SKIP': continue
            elif kind == 'MISMATCH': raise SyntaxError(f"認識できない文字です: '{value}'")
            if kind == 'ID' and value in keywords: kind = value.upper()
            tokens.append((kind, value))
        return cls(tokens)
    def current_token(self) -> Token | None: return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def consume(self, expected_val=None):
        if self.current_token() is None:
            raise SyntaxError(f"入力の終端ですが、'{expected_val}'が予期されました")
        if expected_val and self.current_token()[1] != expected_val:
             raise SyntaxError(f"'{expected_val}'が予期されましたが、'{self.current_token()}'が見つかりました")
        self.pos += 1
    def parse_type(self) -> Type:
        t = self.parse_atomic_type()
        if self.current_token() and self.current_token()[1] == '->':
            self.consume('->')
            return TArrow(t, self.parse_type())
        return t
    def parse_atomic_type(self) -> Type:
        # list修飾子を考慮してprimary_typeから変更
        base_type = self.parse_primary_type()
        if self.current_token() and self.current_token()[1] == 'list':
            self.consume('list')
            return TList(base_type)
        return base_type
    def parse_primary_type(self) -> Type:
        kind, val = self.current_token()
        if val == 'int': self.consume('int'); return TInt()
        if val == 'bool': self.consume('bool'); return TBool()
        if val == '(':
            self.consume('('); t = self.parse_type(); self.consume(')');
            return t
        raise SyntaxError(f"型として予期しないトークンです: '{val}'")
    def parse_expression(self) -> Expression:
        kind, _ = self.current_token() if self.current_token() else (None, None)
        if kind == 'LET': return self.parse_let()
        if kind == 'IF': return self.parse_if()
        if kind == 'FUN': return self.parse_fun()
        if kind == 'MATCH': return self.parse_match()
        return self.parse_comparison()
    def parse_let(self) -> Expression:
        self.consume('let')
        if self.current_token() and self.current_token()[1] == 'rec': return self.parse_let_rec()
        _, var_name = self.current_token(); self.consume(var_name)
        var_type = None
        if self.current_token() and self.current_token()[1] == ':':
            self.consume(':')
            var_type = self.parse_type()
        self.consume('=')
        bound_expr = self.parse_expression(); self.consume('in'); body_expr = self.parse_expression()
        return Let(var_name, bound_expr, body_expr, var_type)
    def parse_let_rec(self) -> Expression:
        self.consume('rec'); _, func_name = self.current_token(); self.consume(func_name); self.consume(':'); func_type = self.parse_type(); self.consume('=')
        self.consume('fun'); self.consume('('); _, arg_name = self.current_token(); self.consume(arg_name); self.consume(':'); arg_type = self.parse_type(); self.consume(')')
        self.consume('->'); func_body = self.parse_expression(); self.consume('in'); let_body = self.parse_expression()
        return LetRec(func_name, func_type, arg_name, arg_type, func_body, let_body)
    def parse_fun(self) -> Expression:
        self.consume('fun')
        has_paren = self.current_token() and self.current_token()[1] == '('
        if has_paren:
            self.consume('(')
        
        _, arg_name = self.current_token()
        self.consume(arg_name)
        
        arg_type = None
        if self.current_token() and self.current_token()[1] == ':':
            self.consume(':')
            arg_type = self.parse_type()

        if has_paren:
            self.consume(')')
            
        self.consume('->')
        body = self.parse_expression()
        return Fun(arg_name, body, arg_type)
    def parse_if(self) -> Expression:
        self.consume('if'); cond = self.parse_expression(); self.consume('then')
        true_branch = self.parse_expression(); self.consume('else'); false_branch = self.parse_expression()
        return If(cond, true_branch, false_branch)
    def parse_match(self) -> Expression:
        self.consume('match'); match_expr = self.parse_expression(); self.consume('with')
        self.consume('[]'); self.consume('->'); nil_branch = self.parse_expression()
        self.consume('|'); _, head_var = self.current_token(); self.consume(head_var); self.consume('::'); _, tail_var = self.current_token(); self.consume(tail_var); self.consume('->')
        cons_branch = self.parse_expression()
        return Match(match_expr, nil_branch, head_var, tail_var, cons_branch)
    def parse_comparison(self) -> Expression:
        node = self.parse_cons()
        if self.current_token() and self.current_token()[1] == '<': self.consume('<'); right_node = self.parse_cons(); node = LessThan(node, right_node)
        return node
    def parse_cons(self) -> Expression:
        node = self.parse_add_sub()
        while self.current_token() and self.current_token()[1] == '::':
            self.consume('::'); right_node = self.parse_cons(); node = Cons(node, right_node)
        return node
    def parse_add_sub(self) -> Expression:
        node = self.parse_mul()
        while self.current_token() and self.current_token()[1] in ['+', '-']:
            _, op_val = self.current_token(); self.consume(op_val); right_node = self.parse_mul(); node = Plus(node, right_node) if op_val == '+' else Minus(node, right_node)
        return node
    def parse_mul(self) -> Expression:
        node = self.parse_app()
        while self.current_token() and self.current_token()[1] == '*': self.consume('*'); right_node = self.parse_app(); node = Times(node, right_node)
        return node
    def parse_app(self) -> Expression:
        node = self.parse_primary()
        while True:
            token = self.current_token()
            if token is None: break
            kind, val = token
            # ifの後に続く式などを考慮し、キーワードでない限り適用とみなす
            if kind in ['INT', 'ID', 'TRUE', 'FALSE'] or val in ['(', '[']: 
                node = App(node, self.parse_primary())
            else: break
        return node
    def parse_primary(self) -> Expression:
        if self.current_token() is None: raise SyntaxError("式の途中で入力が終了しました")
        kind, val = self.current_token()
        if kind == 'INT': self.consume(val); return IntLiteral(int(val))
        if kind == 'ID': self.consume(val); return Variable(val)
        if kind == 'TRUE': self.consume(val); return BoolLiteral(True)
        if kind == 'FALSE': self.consume(val); return BoolLiteral(False)
        if val == '(': self.consume('('); node = self.parse_expression(); self.consume(')'); return node
        if val == '[': self.consume('['); self.consume(']'); return Nil()
        raise SyntaxError(f"予期しないトークンです: '{val}'")

def run_parser_on_text(text: str) -> Expression:
    if not text.strip(): raise SyntaxError("式が空です")
    parser = Parser.from_text(text)
    node = parser.parse_expression()
    if parser.current_token() is not None: raise SyntaxError(f"解析完了後に余分なトークンがあります: '{parser.current_token()}'")
    return node

# -------------------------------------------------------------------
# 6. メイン実行部
# -------------------------------------------------------------------
def parse_tenv_str(tenv_str: str) -> TypeEnv:
    if not tenv_str.strip(): return []
    tenv: TypeEnv = []
    bindings = tenv_str.split(',')
    for binding in bindings:
        parts = binding.split(':')
        if len(parts) != 2: raise SyntaxError(f"型環境の形式が無効です: '{binding}'")
        var_name = parts[0].strip()
        type_str = parts[1].strip()
        
        type_parser = Parser.from_text(type_str)
        var_type = type_parser.parse_type()
        if type_parser.current_token() is not None:
             raise SyntaxError(f"型文字列の解析に失敗しました: '{type_str}'")
        
        tenv.append((var_name, var_type))
    return tenv

def main():
    print("型検査したい式を入力してください。例: |- 1 + 2 : int")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            
            # グローバルな型変数のカウンターをリセット
            TVar._id_counter = 0

            parts = line.split("|-", 1)
            tenv_str, judgment_str = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ("", line)
            
            judgment_parts = judgment_str.rsplit(":", 1)
            expr_str = judgment_parts[0].strip()
            expected_type_str = judgment_parts[1].strip() if len(judgment_parts) > 1 else None
            
            try:
                initial_tenv = parse_tenv_str(tenv_str)
                ast = run_parser_on_text(expr_str)
                
                expected_type = None
                if expected_type_str:
                    type_parser = Parser.from_text(expected_type_str)
                    expected_type = type_parser.parse_type()

                inferred_type, derivation_tree = type_infer(initial_tenv, ast, hint_type=expected_type)
                
                if expected_type:
                    try:
                        unify(inferred_type, expected_type)
                    except TypeError:
                         print(f"型エラー: 期待された型 '{expected_type}' と推論された型 '{resolve(inferred_type)}' が異なります。")

                formatted_tree = derivation_tree.format(0)
                if formatted_tree.endswith(';'): formatted_tree = formatted_tree[:-1]
                print(formatted_tree)
                print("-" * 20)

            except (SyntaxError, TypeError, ValueError, NameError, NotImplementedError) as e:
                print(f"エラー: {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n終了します。")

if __name__ == "__main__":
    main()
