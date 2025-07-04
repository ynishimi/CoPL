|- let s = fun f -> fun g -> fun x -> f x (g x) in let k = fun x -> fun y -> x in s k k : 'a -> 'a by T-Let {
 |- fun f -> fun g -> fun x -> f x (g x) : ('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c by T-Abs {
  f:'a -> 'b -> 'c |- fun g -> fun x -> f x (g x) : ('a -> 'b) -> 'a -> 'c by T-Abs {
   f:'a -> 'b -> 'c, g:'a -> 'b |- fun x -> f x (g x) : 'a -> 'c by T-Abs {
    f:'a -> 'b -> 'c, g:'a -> 'b, x:'a |- f x (g x) : 'c by T-App {
     f:'a -> 'b -> 'c, g:'a -> 'b, x:'a |- f x : 'b -> 'c by T-App {
      f:'a -> 'b -> 'c, g:'a -> 'b, x:'a |- f : 'a -> 'b -> 'c by T-Var {};
      f:'a -> 'b -> 'c, g:'a -> 'b, x:'a |- x : 'a by T-Var {};
     };
     f:'a -> 'b -> 'c, g:'a -> 'b, x:'a |- g x : 'b by T-App {
      f:'a -> 'b -> 'c, g:'a -> 'b, x:'a |- g : 'a -> 'b by T-Var {};
      f:'a -> 'b -> 'c, g:'a -> 'b, x:'a |- x : 'a by T-Var {};
     };
    };
   };
  };
 };
 s:'c 'b 'a.('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c |- let k = fun x -> fun y -> x in s k k : 'a -> 'a by T-Let {
  s:'c 'b 'a.('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c |- fun x -> fun y -> x : 'd -> 'e -> 'd by T-Abs {
   s:'c 'b 'a.('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c, x:'d |- fun y -> x : 'e -> 'd by T-Abs {
    s:'c 'b 'a.('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c, x:'d, y:'e |- x : 'd by T-Var {};
   };
  };
  s:'c 'b 'a.('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c, k:'e 'd.'d -> 'e -> 'd |- s k k : 'a -> 'a by T-App {
   s:'c 'b 'a.('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c, k:'e 'd.'d -> 'e -> 'd |- s k : ('a -> 'f -> 'a) -> 'a -> 'a by T-App {
    s:'c 'b 'a.('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c, k:'e 'd.'d -> 'e -> 'd |- s : ('a -> ('f -> 'a) -> 'a) -> ('a -> 'f -> 'a) -> 'a -> 'a by T-Var {};
    s:'c 'b 'a.('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c, k:'e 'd.'d -> 'e -> 'd |- k : 'a -> ('f -> 'a) -> 'a by T-Var {};
   };
   s:'c 'b 'a.('a -> 'b -> 'c) -> ('a -> 'b) -> 'a -> 'c, k:'e 'd.'d -> 'e -> 'd |- k : 'a -> 'f -> 'a by T-Var {};
  };
 };
}