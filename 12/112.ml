|- let k = fun x -> fun y -> x in k 3 true :: k (1 :: []) 3 : int list by T-Let {
 |- fun x -> fun y -> x : 'a -> 'b -> 'a by T-Abs {
  x:'a |- fun y -> x : 'b -> 'a by T-Abs {
   x:'a, y:'b |- x : 'a by T-Var {};
  };
 };
 k:'b 'a.'a -> 'b -> 'a |- k 3 true :: k (1 :: []) 3 : int list by T-Cons {
  k:'b 'a.'a -> 'b -> 'a |- k 3 true : int by T-App {
   k:'b 'a.'a -> 'b -> 'a |- k 3 : bool -> int by T-App {
    k:'b 'a.'a -> 'b -> 'a |- k : int -> bool -> int by T-Var {};
    k:'b 'a.'a -> 'b -> 'a |- 3 : int by T-Int {};
   };
   k:'b 'a.'a -> 'b -> 'a |- true : bool by T-Bool {};
  };
  k:'b 'a.'a -> 'b -> 'a |- k (1 :: []) 3 : int list by T-App {
   k:'b 'a.'a -> 'b -> 'a |- k (1 :: []) : int -> int list by T-App {
    k:'b 'a.'a -> 'b -> 'a |- k : int list -> int -> int list by T-Var {};
    k:'b 'a.'a -> 'b -> 'a |- 1 :: [] : int list by T-Cons {
     k:'b 'a.'a -> 'b -> 'a |- 1 : int by T-Int {};
     k:'b 'a.'a -> 'b -> 'a |- [] : int list by T-Nil {};
    };
   };
   k:'b 'a.'a -> 'b -> 'a |- 3 : int by T-Int {};
  };
 };
}