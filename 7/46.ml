|- let twice = fun f -> fun x -> f (f x) in twice (fun x -> x * x) 2 evalto 16 by E-Let {
    |- fun f -> fun x -> f (f x) evalto ()[fun f -> fun x -> f (f x)] by E-Fun {};
    twice = ()[fun f -> fun x -> f (f x)] |- twice (fun x -> x * x) 2 evalto 16 by E-App {
        twice = ()[fun f -> fun x -> f (f x)] |- twice (fun x -> x * x) evalto (f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] by E-App {
            twice = ()[fun f -> fun x -> f (f x)] |- twice evalto ()[fun f -> (fun x -> f (f x))] by E-Var1 {};
            twice = ()[fun f -> fun x -> f (f x)] |- fun x -> x * x evalto (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Fun {};
            f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- fun x -> f (f x) evalto (f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x])[fun x -> f (f x)] by E-Fun {};
        };
        twice = ()[fun f -> fun x -> f (f x)] |- 2 evalto 2 by E-Int {};
        f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- f (f x) evalto 16 by E-App {
            f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- f evalto (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var2 {
                f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f evalto (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var1 {};
            };
            f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- f x evalto 4 by E-App {
                f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- f evalto (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var2 {
                    f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x] |- f evalto (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x] by E-Var1 {};
                };
                f = (twice = ()[fun f -> fun x -> f (f x)])[fun x -> x * x], x = 2 |- x evalto 2 by E-Var1 {};
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
};