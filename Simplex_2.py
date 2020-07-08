# DEGENRATE PROBLEMS SOLVING USING SIMPLEX ALGORITHM

import numpy as np  

def convert_to_standard(a,b,c): # Function for converting into standard form
    new_c = []
    for value in c:
        new_c.append(value)
    for i in range(len(b)):
        new_c.append(0)
    c = np.array(new_c)    
    a = np.array(a)
    a = a.astype('float64')
    a = np.hstack((a, np.identity(len(b), dtype = float)))
    return [a,np.array(b),c]

def simplex(A,c, x, basic, rule: int = 0):
    row = A.shape[0] # no. of rows
    col = A.shape[1] # no. of columns of A
    b_I, n_I = list(basic), set(range(col)) - basic  # Basic/nonbasic variables index list
    del basic  
    func = np.dot(c, x)  #Objective function
    B_inv = np.linalg.inv(A[:, b_I])  #Inverse of basic matrix (A[:,B])
    itr = 1
    while True:  # Ensuring termination
        r_q, q, p, theta, d = None, None, None, None, None 
        flag = np.matmul(c[b_I], B_inv) 
        if rule == 0: 
            optimal = True
            for q in n_I:
                r_q = np.asscalar(c[q] - np.matmul(flag,A[:, q]))
                if r_q < 0:
                    optimal = False
                    break  # The loop is exited with the first negative value
        elif rule == 1:
            r_q, q = min([(np.asscalar(c[q] - np.matmul(flag,A[:, q])), q) for q in n_I],
                         key=(lambda tup: tup[0]))
            optimal = (r_q >= 0)
        else:
            raise ValueError("Invalid pivoting")

        if optimal:
            return 0, x, set(b_I),func, None, itr  # Optimal solution

        # Finding feasible basic diection
        d = np.zeros(col)
        for i in range(row):
            d[b_I[i]] = baseline(np.asscalar(np.matmul(-B_inv[i, :],A[:, q])))
        d[q] = 1

        # List of tuples of "candidate" theta an corresponding index in basic variables list:
        neg = [(-x[b_I[i]] / d[b_I[i]], i) for i in range(row) if d[b_I[i]] < 0]

        if len(neg) == 0:
            return 1, x, set(b_I),  None, d, itr  # Problem is Unbounded

        # Get theta and index (in basis):
        theta, p = min(neg, key=(lambda tup: tup[0]))

        # Updating Variables
        x = np.array([baseline(var) for var in (x + theta * d)])  # Update all variables
        func = baseline(func + theta * r_q)  # Objective function value update

        # Update inverse
        for i in set(range(row)) - {p}:
            B_inv[i, :] -= d[b_I[i]]/d[b_I[p]] * B_inv[p, :]
        B_inv[p, :] /= -d[b_I[p]]

        n_I = n_I - {q} | {b_I[p]}  # Non-basic variables index update
        b_I[p] = q  # Update basic index list
        itr += 1
    # If iterations > 500:
    raise TimeoutError("Inumber of iterations exceeds due to endless loop!!!")

# baseline function returns 0 if given input is less then given constant
def baseline(x) -> float:
    return x if abs(x) >= 10**(-10) else 0

def main():
    # Taking input from user
    row = int(input("Enter the number of rows of Matrix A:")) 
    col = int(input("Enter the number of columns of Matrix A:")) 
    if row<col:
        raise ValueError("Number of Variables is more then number of Constraints")
    A = []    # Matirx A of size row*col
    print("Enter the entries in Matrix A rowwise by pressing ENTER after each input:")
    for i in range(row):
        a = []
        for j in range(col):
            a.append(float(input()))
        A.append(a)

    b = []    # Size of vector B should be equal to number of rows in matrix A
    s = int(input("Enter the number of entries in vector B:"))
    if s!=row:
        raise ValueError("Invalid Dimension of vector b")
    for i in range(s):
        b.append(float(input()))

    c = []    # Size of vector C should be equal to number of columns in matrix A
    d = int(input("Enter the number of entries in vector C:"))
    if d!=col:
        raise ValueError("Invalid Dimension of vector c")  
    for i in range(d):
        c.append(-float(input()))

    inp = convert_to_standard(A,b,c)
    A = inp[0]
    b = inp[1]
    c = inp[2]
    rule: int = 0 
    number_of_rows=A.shape[0]
    number_of_columns=A.shape[1]

    if not np.linalg.matrix_rank(A) == number_of_rows:
        # Remove ld rows:
        A = A[[i for i in range(number_of_rows) if not np.array_equal(np.linalg.qr(A)[1][i, :], np.zeros(number_of_columns))], :]
        number_of_rows = A.shape[0]  # Update no. of rows

    A[[i for i in range(number_of_rows) if b[i] < 0]] *= -1  # Change sign of constraints
    b = np.abs(b)  

    temp_C = np.concatenate((np.zeros(number_of_columns), np.ones(number_of_rows))) 
    temp_A = np.matrix(np.concatenate((A, np.identity(number_of_rows)), axis=1))  # Constraint matrix
    variables = np.concatenate((np.zeros(number_of_columns), b))  # Variable vector
    basic_variables = set(range(number_of_columns, number_of_rows + number_of_columns))  # Basic variable set

    whether_optimal, basic_feasible_solution, basis, check_unfeasible, _, it_I = simplex(temp_A, temp_C, variables, basic_variables, rule)
    # Exit code, initial BFS, basis, check_unfeasible, d (not needed) and no. of iterations

    #checking whether problem is feasible or not
    if check_unfeasible > 0:
        print("Problem is infeasible")
        return

    basic_feasible_solution = basic_feasible_solution[:number_of_columns]
    ext, x, basic, z, d, it_II = simplex(A, c, basic_feasible_solution, basis, rule)
    if ext == 0:
        print("Optimal Solution Found at x = "+str(x))
        print("Optimal Solution : "+str(-z))
    elif ext == 1:
        print("Problem is unbounded")    

    return 
    
if __name__ == '__main__':
    main()    
