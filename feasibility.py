def check_feasibility(idea):
    # Simple heuristic for feasibility
    if "advanced hardware" in idea.lower():
        print("Not Feasible: Requires advanced hardware.")
        return "Not Feasible: Requires advanced hardware."
    elif "high cost" in idea.lower():
        print( "Not Feasible: Budget constraints.")
        return "Not Feasible: Budget constraints."
    else:
        return "Feasible: Idea seems realistic with current technologies."
