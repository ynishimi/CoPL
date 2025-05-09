x = true, y = 4 |- if x then y + 1 else y - 1 evalto 5 by E-IfT {
    // |- の直前にないものはそのままでは使えず、 E-Var2 を使って開く必要がある
    x = true, y = 4 |- x evalto true by E-Var2 {
        x = true |- x evalto true by E-Var1 {};
    };
    x = true, y = 4 |- y + 1 evalto 5 by E-Plus {
        x = true, y = 4 |- y evalto 4 by E-Var1 {};
        x = true, y = 4 |- 1 evalto 1 by E-Int {};
        4 plus 1 is 5 by B-Plus {};
    };
};