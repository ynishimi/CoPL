|- let x = (let y = 3 - 2 in y * y) in (let y = 4 in x + y) evalto 5 by E-Let {
    |- let y = 3 - 2 in y * y evalto 1 by E-Let {
        |- 3 - 2 evalto 1 by E-Minus {
            |- 3 evalto 3 by E-Int {};
            |- 2 evalto 2 by E-Int {};
            3 minus 2 is 1 by B-Minus {};
        };
        y = 1 |- y * y evalto 1 by E-Times {
            y = 1 |- y evalto 1 by E-Var1 {};
            y = 1 |- y evalto 1 by E-Var1 {};
            1 times 1 is 1 by B-Times {};
        };
    };
    x = 1 |- let y = 4 in x + y evalto 5 by E-Let {
        x = 1 |- 4 evalto 4 by E-Int {};
        x = 1, y = 4 |- x + y evalto 5 by E-Plus {
            x = 1, y = 4 |- x evalto 1 by E-Var2 {
                x = 1 |- x evalto 1 by E-Var1 {};
            };
            x = 1, y = 4 |- y evalto 4 by E-Var1 {};
            1 plus 4 is 5 by B-Plus {};
        };
    };
};