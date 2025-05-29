|- let max = fun x -> fun y -> if x < y then y else x in max 3 5 evalto 5 by E-Let {
    |- fun x -> fun y -> if x < y then y else x evalto ()[fun x -> fun y -> if x < y then y else x] by E-Fun {};
    max = ()[fun x -> fun y -> if x < y then y else x] |- max 3 5 evalto 5 by E-App {

    };
};