# polytypingml4.py
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
        p_type_resolved = resolve(self.param_type)
        p_str = str(self.param_type)
        if isinstance(p_type_resolved, TArrow):
            p_str = f"({p_str})"
        return f"{p_str} -> {self.return_type}"

@dataclass(frozen=True)
class TList(Type):
    element_type: Type
    def __str__(self):
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
    def __init__(self, from_snapshot=False):
        if not from_snapshot:
            self.id = TVar._id_counter
            TVar._id_counter += 1
        else:
            self.id = -1
        self.instance: Optional[Type] = None
        self.display_name: Optional[str] = None

    def __str__(self):
        if self.instance:
            return str(resolve(self.instance))
        if self.display_name:
            return self.display_name
        return f"'t{self.id}"

    # MODIFICATION: Add id-based equality and hashing for TVar
    def __eq__(self, other):
        if not isinstance(other, TVar):
            return NotImplemented
        # Two type variables are equal if they are the same variable (identified by id).
        return self.id == other.id

    def __hash__(self):
        # Hash based on the unique id.
        return hash(self.id)

@dataclass
class TypeScheme:
    """let多相のための型スキームを表すクラス。"""
    variables: List[TVar]
    body: Type
    def __str__(self):
        if not self.variables:
            return str(self.body)
        
        mapping = {}
        snap_vars = [snapshot_type(v, mapping) for v in self.variables]
        snap_body = snapshot_type(self.body, mapping)

        var_str = " ".join(str(v) for v in snap_vars)
        return f"{var_str}.{snap_body}"


# -------------------------------------------------------------------
# 2. 式の構造 (AST)
# -------------------------------------------------------------------
class Expression:
    def to_string(self) -> str: return str(self)
    def get_precedence(self): return 99

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
    var: str
    bound_expr: Expression # Must be a Fun expression
    body_expr: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return 0
    def to_string(self):
        return f"let rec {self.var} = {self.bound_expr.to_string()} in {self.body_expr.to_string()}"

@dataclass
class Match(Expression):
    match_expr: Expression; nil_branch: Expression; head_var: str; tail_var: str; cons_branch: Expression
    def __str__(self): return self.to_string()
    def get_precedence(self): return precedence.get(self.__class__.__name__, 0)
    def to_string(self): return f"match {self.match_expr.to_string()} with [] -> {self.nil_branch.to_string()} | {self.head_var} :: {self.tail_var} -> {self.cons_branch.to_string()}"

TypeEnv = List[Tuple[str, Union[Type, TypeScheme]]]
Token = Tuple[str, str]

# -------------------------------------------------------------------
# 3. 導出規則 (Derivation)
# -------------------------------------------------------------------
def format_tenv_display(tenv: List[Tuple[str, Union[Type, TypeScheme]]]) -> str:
    if not tenv: return ""
    return ", ".join(f"{var}:{t}" for var, t in tenv) + " "

class Derivation:
    def format(self, i: int = 0, mapping: Optional[dict] = None) -> str: raise NotImplementedError

@dataclass
class TGeneric(Derivation):
    tenv: TypeEnv
    expr: Expression
    type: Type
    rule_name: str
    premises: List[Derivation]

    def format(self, i: int = 0, mapping: Optional[dict] = None) -> str:
        if mapping is None:
            mapping = {}

        indent = " " * i
        
        display_type = snapshot_type(self.type, mapping)
        display_tenv = snapshot_tenv(self.tenv, mapping)
        tenv_str = format_tenv_display(display_tenv)
        
        if not self.premises:
            return f"{indent}{tenv_str}|- {self.expr.to_string()} : {display_type} by {self.rule_name} {{}};"
        else:
            premise_str = "\n".join(p.format(i + 1, mapping) for p in self.premises)
            return f"{indent}{tenv_str}|- {self.expr.to_string()} : {display_type} by {self.rule_name} {{\n{premise_str}\n{indent}}};"

# -------------------------------------------------------------------
# 4. 型推論エンジン (Unification-based Type Inference with Let-Polymorphism)
# -------------------------------------------------------------------

def resolve(t: Type) -> Type:
    if isinstance(t, TVar) and t.instance:
        t.instance = resolve(t.instance)
        return t.instance
    return t

def snapshot_type(t: Type, mapping: Optional[dict] = None) -> Type:
    if mapping is None:
        mapping = {}
    t = resolve(t)
    if id(t) in mapping:
        return mapping[id(t)]
    
    if isinstance(t, TVar):
        frozen_var = TVar(from_snapshot=True)
        frozen_var.id = t.id
        frozen_var.display_name = t.display_name
        mapping[id(t)] = frozen_var
        if t.instance:
            frozen_var.instance = snapshot_type(t.instance, mapping)
        return frozen_var
    elif isinstance(t, TArrow):
        new_arrow = TArrow(snapshot_type(t.param_type, mapping), snapshot_type(t.return_type, mapping))
        mapping[id(t)] = new_arrow
        return new_arrow
    elif isinstance(t, TList):
        new_list = TList(snapshot_type(t.element_type, mapping))
        mapping[id(t)] = new_list
        return new_list
    else:
        mapping[id(t)] = t
        return t

def snapshot_tenv(tenv: TypeEnv, mapping: Optional[dict] = None) -> List[Tuple[str, Union[Type, TypeScheme]]]:
    if mapping is None:
        mapping = {}
    new_tenv = []
    for var, t_or_s in tenv:
        if isinstance(t_or_s, TypeScheme):
            snap_vars = [snapshot_type(v, mapping) for v in t_or_s.variables]
            snap_body = snapshot_type(t_or_s.body, mapping)
            new_tenv.append((var, TypeScheme(snap_vars, snap_body)))
        else:
            new_tenv.append((var, snapshot_type(t_or_s, mapping)))
    return new_tenv

def get_free_vars(t: Union[Type, TypeEnv, TypeScheme]) -> set[TVar]:
    free_vars = set()
    def collector(ty: Type):
        ty = resolve(ty)
        if isinstance(ty, TVar):
            free_vars.add(ty)
        elif isinstance(ty, TArrow):
            collector(ty.param_type)
            collector(ty.return_type)
        elif isinstance(ty, TList):
            collector(ty.element_type)

    if isinstance(t, list):
        for _, t_or_s in t:
            free_vars.update(get_free_vars(t_or_s))
    elif isinstance(t, TypeScheme):
        free_vars.update(get_free_vars(t.body))
        free_vars.difference_update(t.variables)
    else:
        collector(t)
    return free_vars

def generalize(tenv: TypeEnv, t: Type) -> TypeScheme:
    env_vars = get_free_vars(tenv)
    type_vars = get_free_vars(t)
    quantify_vars = list(type_vars - env_vars)
    return TypeScheme(quantify_vars, t)

def instantiate(scheme: TypeScheme) -> Type:
    substitutions = {}
    for v in scheme.variables:
        new_var = TVar()
        new_var.display_name = v.display_name
        substitutions[v.id] = new_var

    def instantiator(t: Type) -> Type:
        t = resolve(t)
        if isinstance(t, TVar) and t.id in substitutions:
            return substitutions[t.id]
        elif isinstance(t, TArrow):
            return TArrow(instantiator(t.param_type), instantiator(t.return_type))
        elif isinstance(t, TList):
            return TList(instantiator(t.element_type))
        else:
            return t
    return instantiator(scheme.body)

def unify(t1: Type, t2: Type):
    t1 = resolve(t1)
    t2 = resolve(t2)
    if isinstance(t1, TVar):
        if t1 == t2: return
        if t1 in get_free_vars(t2):
            raise TypeError(f"無限の型を検出しました: {t1} in {t2}")
        t1.instance = t2
        return
    if isinstance(t2, TVar):
        if t2 in get_free_vars(t1):
            raise TypeError(f"無限の型を検出しました: {t2} in {t1}")
        t2.instance = t1
        return
    if isinstance(t1, TArrow) and isinstance(t2, TArrow):
        unify(t1.param_type, t2.param_type)
        unify(t1.return_type, t2.return_type)
        return
    if isinstance(t1, TList) and isinstance(t2, TList):
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
        for var, t_or_scheme in reversed(tenv):
            if var == expr.name:
                t = instantiate(t_or_scheme) if isinstance(t_or_scheme, TypeScheme) else t_or_scheme
                return t, TGeneric(tenv, expr, t, "T-Var", [])
        raise TypeError(f"未定義の変数です: {expr.name}")

    if isinstance(expr, BinaryOp):
        if expr.op == '::':
            t1, d1 = type_infer(tenv, expr.left)
            hint_for_right = TList(t1)
            t2, d2 = type_infer(tenv, expr.right, hint_type=hint_for_right)
            unify(t2, hint_for_right)
            return resolve(t2), TGeneric(tenv, expr, resolve(t2), "T-Cons", [d1, d2])
            
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
        scheme = generalize(tenv, t1)
        t2, d2 = type_infer(tenv + [(expr.var, scheme)], expr.body_expr, hint_type=hint_type)
        return t2, TGeneric(tenv, expr, t2, "T-Let", [d1, d2])

    if isinstance(expr, Fun):
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
        return func_type, TGeneric(tenv, expr, func_type, "T-Abs", [d_body])

    if isinstance(expr, App):
        t_func, d_func = type_infer(tenv, expr.func_expr)
        t_arg, d_arg = type_infer(tenv, expr.arg_expr)
        ret_type = TVar()
        expected_func_type = TArrow(t_arg, ret_type)
        unify(t_func, expected_func_type)
        final_ret_type = resolve(ret_type)
        return final_ret_type, TGeneric(tenv, expr, final_ret_type, "T-App", [d_func, d_arg])

    if isinstance(expr, LetRec):
        if not isinstance(expr.bound_expr, Fun):
            raise TypeError("let recの右辺は関数である必要があります")
        
        new_ty = TVar()
        env_with_rec_var = tenv + [(expr.var, new_ty)]
        inferred_body_type, d1 = type_infer(env_with_rec_var, expr.bound_expr)
        unify(new_ty, inferred_body_type)
        scheme = generalize(tenv, new_ty)
        env_for_in_part = tenv + [(expr.var, scheme)]
        final_type, d2 = type_infer(env_for_in_part, expr.body_expr, hint_type=hint_type)
        
        return final_type, TGeneric(tenv, expr, final_type, "T-LetRec", [d1, d2])

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
class Parser:
    def __init__(self, tokens: List[Token]): self.tokens, self.pos = tokens, 0
    @classmethod
    def from_text(cls, text: str):
        keywords = {'if', 'then', 'else', 'true', 'false', 'let', 'in', 'fun', 'rec', 'match', 'with', 'int', 'bool', 'list'}
        token_spec = [('T_VAR', r"'[a-zA-Z_][a-zA-Z0-9_]*"), ('CONS', r'::'), ('ARROW', r'->'), ('BAR', r'\|'), ('DOT', r'\.'), ('ID', r'[a-zA-Z_][a-zA-Z0-9_]*'), ('INT', r'-?\d+'), ('OP', r'[+*<()=\[\]_:-]'), ('SKIP', r'\s+'), ('MISMATCH', r'.')]
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
        
    def parse_scheme(self) -> Union[Type, TypeScheme]:
        type_vars_context = {}
        quantified_vars = []
        
        temp_pos = self.pos
        while temp_pos < len(self.tokens) and self.tokens[temp_pos][0] == 'T_VAR':
            temp_pos += 1
        
        if temp_pos < len(self.tokens) and self.tokens[temp_pos][0] == 'DOT':
            while self.current_token() and self.current_token()[0] == 'T_VAR':
                kind, val = self.current_token()
                self.consume(val)
                if val not in type_vars_context:
                    new_tvar = TVar()
                    new_tvar.display_name = val
                    type_vars_context[val] = new_tvar
                quantified_vars.append(type_vars_context[val])
            
            self.consume('.')
            body = self.parse_type(type_vars_context)
            return TypeScheme(quantified_vars, body)
        else:
            return self.parse_type(type_vars_context)

    def parse_type(self, type_vars_context: Optional[dict] = None) -> Type:
        if type_vars_context is None:
            type_vars_context = {}
        
        t = self.parse_atomic_type(type_vars_context)
        if self.current_token() and self.current_token()[1] == '->':
            self.consume('->')
            return TArrow(t, self.parse_type(type_vars_context))
        return t

    def parse_atomic_type(self, type_vars_context: dict) -> Type:
        base_type = self.parse_primary_type(type_vars_context)
        if self.current_token() and self.current_token()[1] == 'list':
            self.consume('list')
            return TList(base_type)
        return base_type

    def parse_primary_type(self, type_vars_context: dict) -> Type:
        kind, val = self.current_token()
        if kind == 'T_VAR':
            self.consume(val)
            if val not in type_vars_context:
                new_tvar = TVar()
                new_tvar.display_name = val
                type_vars_context[val] = new_tvar
            return type_vars_context[val]
        if val == 'int': self.consume('int'); return TInt()
        if val == 'bool': self.consume('bool'); return TBool()
        if val == '(':
            self.consume('(')
            t = self.parse_type(type_vars_context)
            self.consume(')')
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
        if self.current_token() and self.current_token()[1] == 'rec':
            return self.parse_let_rec()
        _, var_name = self.current_token(); self.consume(var_name)
        var_type = None
        if self.current_token() and self.current_token()[1] == ':':
            self.consume(':')
            var_type = self.parse_type()
        self.consume('=')
        bound_expr = self.parse_expression(); self.consume('in'); body_expr = self.parse_expression()
        return Let(var_name, bound_expr, body_expr, var_type)
    def parse_let_rec(self) -> Expression:
        self.consume('rec')
        _, var_name = self.current_token(); self.consume(var_name)
        self.consume('=')
        bound_expr = self.parse_expression()
        if not isinstance(bound_expr, Fun):
            raise SyntaxError("let recの右辺は関数リテラルである必要があります")
        self.consume('in')
        body_expr = self.parse_expression()
        return LetRec(var_name, bound_expr, body_expr)
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
        self.consume('['); self.consume(']'); self.consume('->'); nil_branch = self.parse_expression()
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
            self.consume('::'); right_node = self.parse_add_sub(); node = Cons(node, right_node)
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

def canonicalize_derivation_names(derivation: Derivation):
    """
    導出木全体を走査し、一時的な型変数に 'a, 'b, 'c のような名前を付ける。
    これにより、型推論の出力が読みやすくなる。
    """
    all_tvars = set()
    q: List[Derivation] = [derivation]
    visited = set()

    while q:
        d = q.pop(0)
        if id(d) in visited: continue
        visited.add(id(d))

        if isinstance(d, TGeneric):
            all_tvars.update(get_free_vars(d.type))
            all_tvars.update(get_free_vars(d.tenv))
            
            for p in reversed(d.premises):
                q.insert(0, p)

    vars_to_rename = sorted(
        [v for v in all_tvars if v.display_name is None],
        key=lambda v: v.id
    )

    name_counter = 0
    existing_names = {v.display_name for v in all_tvars if v.display_name}

    for tvar in vars_to_rename:
        if tvar.display_name is not None: continue
        while True:
            new_name = f"'{chr(ord('a') + name_counter)}"
            name_counter += 1
            if new_name not in existing_names:
                tvar.display_name = new_name
                existing_names.add(new_name)
                break

def parse_tenv_str(tenv_str: str) -> TypeEnv:
    if not tenv_str.strip(): return []
    tenv: TypeEnv = []
    bindings = tenv_str.split(',')
    for binding in bindings:
        parts = binding.split(':', 1)
        if len(parts) != 2: raise SyntaxError(f"型環境の形式が無効です: '{binding}'")
        var_name = parts[0].strip()
        type_str = parts[1].strip()
        
        type_parser = Parser.from_text(type_str)
        scheme_or_type = type_parser.parse_scheme()
        
        if type_parser.current_token() is not None:
             raise SyntaxError(f"型文字列の解析に失敗しました: '{type_str}'")
        
        tenv.append((var_name, scheme_or_type))
            
    return tenv

def main():
    print("型検査したい式を入力してください。例: |- 1 + 2 : int")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            
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
                
                canonicalize_derivation_names(derivation_tree)
                
                is_poly_annotation = False
                if expected_type:
                    if get_free_vars(expected_type):
                        is_poly_annotation = True
                    
                    try:
                        unify(inferred_type, expected_type)
                    except TypeError:
                         print(f"型エラー: 期待された型 '{expected_type}' と推論された型 '{resolve(inferred_type)}' が異なります。")
                         continue

                if not is_poly_annotation:
                    final_free_vars = get_free_vars(inferred_type)
                    for tvar in final_free_vars:
                        if tvar.instance is None:
                            tvar.instance = TBool()

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