|- let x = 3 * 3 in let y = 4 * x in x + y evalto 45 by E-Let {
    |- 3 * 3 evalto 9 by E-Times {
        |- 3 evalto 3 by E-Int {};
        |- 3 evalto 3 by E-Int {};
        3 times 3 is 9 by B-Times {};
    };
    x = 9 |- let y = 4 * x in x + y evalto 45 by E-Let {
        x = 9 |- 4 * x evalto 36 by E-Times {
            x = 9 |- 4 evalto 4 by E-Int {};
            x = 9 |- x evalto 9 by E-Var1 {};
            4 times 9 is 36 by B-Times {};
        };
        x = 9, y = 36 |- x + y evalto 45 by E-Plus {
            x = 9, y = 36 |- x evalto 9 by E-Var2 {
                x = 9 |- x evalto 9 by E-Var1  {};
            };
            x = 9, y = 36 |- y evalto 36 by E-Var1 {};
            9 plus 36 is 45 by B-Plus {};
       };
    };
};