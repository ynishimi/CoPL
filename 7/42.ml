|- let sq = fun x -> x * x in sq 3 + sq 4 evalto 25 by E-Let {
    |- fun x -> x * x evalto ()[fun x -> x * x] by E-Fun {};
    sq = ()[fun x -> x * x] |- sq 3 + sq 4 evalto 25 by E-Plus {
        sq = ()[fun x -> x * x] |- sq 3 evalto 9 by E-App {
            sq = ()[fun x -> x * x] |- sq evalto ()[fun x -> x * x] by E-Var1 {};
            sq = ()[fun x -> x * x] |- 3 evalto 3 by E-Int {};
            x = 3 |- x * x evalto 9 by E-Times {
                x = 3 |- x evalto 3 by E-Var1 {};
                x = 3 |- x evalto 3 by E-Var1 {};
                3 times 3 is 9 by B-Times {};
            };
        };
        sq = ()[fun x -> x * x] |- sq 4 evalto 16 by E-App {
            sq = ()[fun x -> x * x] |- sq evalto ()[fun x -> x * x] by E-Var1 {};
            sq = ()[fun x -> x * x] |- 4 evalto 4 by E-Int {};
            x = 4 |- x * x evalto 16 by E-Times {
                x = 4 |- x evalto 4 by E-Var1 {};
                x = 4 |- x evalto 4 by E-Var1 {};
                4 times 4 is 16 by B-Times {};
            };
        };
        9 plus 16 is 25 by B-Plus {};
    }
};