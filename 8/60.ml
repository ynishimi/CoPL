|- let x = let y = (3 - 2) in (y * y) in let y = 4 in (x + y) ==> let . = let . = 3 - 2 in #1 * #1 in let . = 4 in #2 + #1 by Tr-Let {
 |- let y = (3 - 2) in (y * y) ==> let . = 3 - 2 in #1 * #1 by Tr-Let {
  |- (3 - 2) ==> 3 - 2 by Tr-Minus {
   |- 3 ==> 3 by Tr-Int {};
   |- 2 ==> 2 by Tr-Int {};
  };
  y |- (y * y) ==> #1 * #1 by Tr-Times {
   y |- y ==> #1 by Tr-Var1 {};
   y |- y ==> #1 by Tr-Var1 {};
  };
 };
 x |- let y = 4 in (x + y) ==> let . = 4 in #2 + #1 by Tr-Let {
  x |- 4 ==> 4 by Tr-Int {};
  x, y |- (x + y) ==> #2 + #1 by Tr-Plus {
   x, y |- x ==> #2 by Tr-Var2 {
    x |- x ==> #1 by Tr-Var1 {};
   };
   x, y |- y ==> #1 by Tr-Var1 {};
  };
 };
}