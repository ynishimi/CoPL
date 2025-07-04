|- let f = fun x -> let g = fun y -> y x :: [] in g (fun z -> 4) in match f true with [] -> 3 :: [] | x :: y -> f x : int list by T-Let {
 |- fun x -> let g = fun y -> y x :: [] in g (fun z -> 4) : 'b -> int list by T-Abs {
  x:'b |- let g = fun y -> y x :: [] in g (fun z -> 4) : int list by T-Let {
   x:'b |- fun y -> y x :: [] : ('b -> 'a) -> 'a list by T-Abs {
    x:'b, y:'b -> 'a |- y x :: [] : 'a list by T-Cons {
     x:'b, y:'b -> 'a |- y x : 'a by T-App {
      x:'b, y:'b -> 'a |- y : 'b -> 'a by T-Var {};
      x:'b, y:'b -> 'a |- x : 'b by T-Var {};
     };
     x:'b, y:'b -> 'a |- [] : 'a list by T-Nil {};
    };
   };
   x:'b, g:'a.('b -> 'a) -> 'a list |- g (fun z -> 4) : int list by T-App {
    x:'b, g:'a.('b -> 'a) -> 'a list |- g : ('b -> int) -> int list by T-Var {};
    x:'b, g:'a.('b -> 'a) -> 'a list |- fun z -> 4 : 'b -> int by T-Abs {
     x:'b, g:'a.('b -> 'a) -> 'a list, z:'b |- 4 : int by T-Int {};
    };
   };
  };
 };
 f:'b.'b -> int list |- match f true with [] -> 3 :: [] | x :: y -> f x : int list by T-Match {
  f:'b.'b -> int list |- f true : int list by T-App {
   f:'b.'b -> int list |- f : bool -> int list by T-Var {};
   f:'b.'b -> int list |- true : bool by T-Bool {};
  };
  f:'b.'b -> int list |- 3 :: [] : int list by T-Cons {
   f:'b.'b -> int list |- 3 : int by T-Int {};
   f:'b.'b -> int list |- [] : int list by T-Nil {};
  };
  f:'b.'b -> int list, x:int, y:int list |- f x : int list by T-App {
   f:'b.'b -> int list, x:int, y:int list |- f : int -> int list by T-Var {};
   f:'b.'b -> int list, x:int, y:int list |- x : int by T-Var {};
  };
 };
}