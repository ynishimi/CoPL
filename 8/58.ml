x |- let x = (x * 2) in (x + x) ==> let . = #1 * 2 in #1 + #1 by Tr-Let {
 x |- (x * 2) ==> #1 * 2 by Tr-Times {
  x |- x ==> #1 by Tr-Var1 {};
  x |- 2 ==> 2 by Tr-Int {};
 };
 x, x |- (x + x) ==> #1 + #1 by Tr-Plus {
  x, x |- x ==> #1 by Tr-Var1 {};
  x, x |- x ==> #1 by Tr-Var1 {};
 };
}