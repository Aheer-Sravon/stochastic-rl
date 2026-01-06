import gurobipy as gp
from gurobipy import GRB

m = gp.Model()

x = m.addVar(lb=0, name="x")
y = m.addVar(lb=0, name="y")

m.setObjective(3*x +2*y, GRB.MAXIMIZE)
m.addConstr(x + y <= 4)

m.optimize()

for v in m.getVars():
    print(v.VarName, v.x)

print(f"Objective: {m.ObjVal}")
