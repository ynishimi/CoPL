x, y |- if x then (y + 1) else (y - 1) ==> if #2 then #1 + 1 else #1 - 1 by Tr-If {
 x, y |- x ==> #2 by Tr-Var2 {
  x |- x ==> #1 by Tr-Var1 {};
 };
 x, y |- (y + 1) ==> #1 + 1 by Tr-Plus {
  x, y |- y ==> #1 by Tr-Var1 {};
  x, y |- 1 ==> 1 by Tr-Int {};
 };
 x, y |- (y - 1) ==> #1 - 1 by Tr-Minus {
  x, y |- y ==> #1 by Tr-Var1 {};
  x, y |- 1 ==> 1 by Tr-Int {};
 };
}