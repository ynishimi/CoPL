|- let twice = fun f -> fun x -> f (f x) in twice twice (fun x -> x * x) 2 evalto 65536 by E-Let {
 |- fun f -> fun x -> f (f x) evalto ()[fun f -> fun x -> f (f x)] by E-Fun {};
 twice = ()[fun f -> fun x -> f (f x)] |- twice twice (fun x -> x * x) 2 evalto 65536 by E-App {
  twice = ()[fun f -> fun x -> f (f x)] |- twice twice (fun x -> x * x) evalto (f=(f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)])[fun x -> f (f x)] by E-App {
   twice = ()[fun f -> fun x -> f (f x)] |- twice twice evalto (f=()[fun f -> fun x -> f (f x)])[fun x -> f (f x)] by E-App {
    twice = ()[fun f -> fun x -> f (f x)] |- twice evalto ()[fun f -> fun x -> f (f x)] by E-Var1 {};
    twice = ()[fun f -> fun x -> f (f x)] |- twice evalto ()[fun f -> fun x -> f (f x)] by E-Var1 {};
    f = ()[fun f -> fun x -> f (f x)] |- fun x -> f (f x) evalto (f=()[fun f -> fun x -> f (f x)])[fun x -> f (f x)] by E-Fun {};
   };
   twice = ()[fun f -> fun x -> f (f x)] |- fun x -> x * x evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Fun {};
   f = ()[fun f -> fun x -> f (f x)], x = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f (f x) evalto (f=(f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)])[fun x -> f (f x)] by E-App {
    f = ()[fun f -> fun x -> f (f x)], x = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f evalto ()[fun f -> fun x -> f (f x)] by E-Var2 {
     f = ()[fun f -> fun x -> f (f x)] |- f evalto ()[fun f -> fun x -> f (f x)] by E-Var1 {};
    };
    f = ()[fun f -> fun x -> f (f x)], x = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f x evalto (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] by E-App {
     f = ()[fun f -> fun x -> f (f x)], x = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f evalto ()[fun f -> fun x -> f (f x)] by E-Var2 {
      f = ()[fun f -> fun x -> f (f x)] |- f evalto ()[fun f -> fun x -> f (f x)] by E-Var1 {};
     };
     f = ()[fun f -> fun x -> f (f x)], x = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- x evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var1 {};
     f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- fun x -> f (f x) evalto (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] by E-Fun {};
    };
    f = (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] |- fun x -> f (f x) evalto (f=(f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)])[fun x -> f (f x)] by E-Fun {};
   };
  };
  twice = ()[fun f -> fun x -> f (f x)] |- 2 evalto 2 by E-Int {};
  f = (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)], x = 2 |- f (f x) evalto 65536 by E-App {
   f = (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)], x = 2 |- f evalto (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] by E-Var2 {
    f = (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] |- f evalto (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] by E-Var1 {};
   };
   f = (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)], x = 2 |- f x evalto 16 by E-App {
    f = (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)], x = 2 |- f evalto (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] by E-Var2 {
     f = (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] |- f evalto (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] by E-Var1 {};
    };
    f = (f=(twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)], x = 2 |- x evalto 2 by E-Var1 {};
    f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- f (f x) evalto 16 by E-App {
     f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- f evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var2 {
      f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var1 {};
     };
     f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- f x evalto 4 by E-App {
      f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- f evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var2 {
       f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var1 {};
      };
      f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- x evalto 2 by E-Var1 {};
      twice = ()[fun f -> fun x -> f (f x)], x = 2 |- x * x evalto 4 by E-Times {
       twice = ()[fun f -> fun x -> f (f x)], x = 2 |- x evalto 2 by E-Var1 {};
       twice = ()[fun f -> fun x -> f (f x)], x = 2 |- x evalto 2 by E-Var1 {};
       2 times 2 is 4 by B-Times {};
      };
     };
     twice = ()[fun f -> fun x -> f (f x)], x = 4 |- x * x evalto 16 by E-Times {
      twice = ()[fun f -> fun x -> f (f x)], x = 4 |- x evalto 4 by E-Var1 {};
      twice = ()[fun f -> fun x -> f (f x)], x = 4 |- x evalto 4 by E-Var1 {};
      4 times 4 is 16 by B-Times {};
     };
    };
   };
   f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 16 |- f (f x) evalto 65536 by E-App {
    f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 16 |- f evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var2 {
     f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var1 {};
    };
    f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 16 |- f x evalto 256 by E-App {
     f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 16 |- f evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var2 {
      f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f evalto (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var1 {};
     };
     f = (twice=()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 16 |- x evalto 16 by E-Var1 {};
     twice = ()[fun f -> fun x -> f (f x)], x = 16 |- x * x evalto 256 by E-Times {
      twice = ()[fun f -> fun x -> f (f x)], x = 16 |- x evalto 16 by E-Var1 {};
      twice = ()[fun f -> fun x -> f (f x)], x = 16 |- x evalto 16 by E-Var1 {};
      16 times 16 is 256 by B-Times {};
     };
    };
    twice = ()[fun f -> fun x -> f (f x)], x = 256 |- x * x evalto 65536 by E-Times {
     twice = ()[fun f -> fun x -> f (f x)], x = 256 |- x evalto 256 by E-Var1 {};
     twice = ()[fun f -> fun x -> f (f x)], x = 256 |- x evalto 256 by E-Var1 {};
     256 times 256 is 65536 by B-Times {};
    };
   };
  };
 };
}