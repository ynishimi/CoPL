|- let x = (3 * 3) in let y = (4 * x) in (x + y) ==> let . = 3 * 3 in let . = 4 * #1 in #2 + #1 by Tr-Let {
 |- (3 * 3) ==> 3 * 3 by Tr-Times {
  |- 3 ==> 3 by Tr-Int {};
  |- 3 ==> 3 by Tr-Int {};
 };
 x |- let y = (4 * x) in (x + y) ==> let . = 4 * #1 in #2 + #1 by Tr-Let {
  x |- (4 * x) ==> 4 * #1 by Tr-Times {
   x |- 4 ==> 4 by Tr-Int {};
   x |- x ==> #1 by Tr-Var1 {};
  };
  x, y |- (x + y) ==> #2 + #1 by Tr-Plus {
   x, y |- x ==> #2 by Tr-Var2 {
    x |- x ==> #1 by Tr-Var1 {};
   };
   x, y |- y ==> #1 by Tr-Var1 {};
  };
 };
}