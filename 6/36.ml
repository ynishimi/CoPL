|- let x = 1 + 2 in x * 4 evalto 12 by E-Let {
    |- 1 + 2 evalto 3 by E-Plus {
        |- 1 evalto 1 by E-Int {};
        |- 2 evalto 2 by E-Int {};
        1 plus 2 is 3 by B-Plus {};
    };
    x = 3 |- x * 4 evalto 12 by E-Times {
        x = 3 |- x evalto 3 by E-Var1 {};
        x = 3 |- 4 evalto 4 by E-Int {};
        3 times 4 is 12 by B-Times {};
    };
};