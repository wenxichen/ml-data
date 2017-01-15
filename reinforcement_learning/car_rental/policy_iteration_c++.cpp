/* 
 This is an c++ implementation of policy iteration as the solution for Example 4.2 (page 87 in second edition)
 in the book "Reinforcement Learning: An Introduction" by Sutton & Barto.
 The code is derived from Daniel Martí's cweb code of the same example.
 The cweb code can be found at http://lumiere.ens.fr/~dmarti01/software/jack.w
 Daniel Martí's website http://lumiere.ens.fr/~dmarti01/software/
 The describtion of the example https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node43.html
 */

#include <stdio.h>
#include <math.h>
#include <algorithm>
using namespace std; /* for cleaner code (min and max functions) */

const int ncar_states = 21;
const int max_moves = 5;
const int max_morning = ncar_states + max_moves ;
const double discount = 0.9;
const double theta = 1e-7; /* stop when differences are of order theta */

double prob_1[max_morning][ncar_states]; /* 26 x 21 */
double prob_2[max_morning][ncar_states];

/* element rew_1[n1] contains the expected immediate reward due to satisfied requests at location 1,
 given that the day starts with n1 cars at location 1 */
double rew_1 [max_morning];
double rew_2 [max_morning];

double V[ncar_states][ncar_states];
int policy[ncar_states][ncar_states];

double factorial(int n);
double poisson(int n, double l);
void load_probs_rewards (double probs[max_morning][ncar_states], double rewards[max_morning],
                         double l_reqsts, double l_drpffs);
void policy_eval();
double backup_action(int n1, int n2, int a);
int greedy_policy (int n1 , int n2);
bool update_policy_t();


double factorial(int n)
{
    if (n > 0) return (n * factorial(n - 1));
    else return (1.0);
}


double poisson(int n, double lambda )
{
    return (exp(-lambda ) * pow(lambda ,(double) n)/factorial(n));
}


void load_probs_rewards (double probs[max_morning][ncar_states], double rewards[max_morning],
                         double l_reqsts, double l_drpffs)
{
    /* 
     Calculate and load the transition probability from morning state to end-of-the-day state
     and the expected rewards of each morning state. 
     */
    
    double req_prob ;
    double drp_prob ;
    int satisfied_req ;
    int new_n;
    for (int req = 0; (req_prob = poisson(req, l_reqsts)) > theta ; req ++) {
        for (int n = 0; n < max_morning ; n++) {
            /* 
             There is an upper limit in the amount of reward received from requests.
             This limit is given by the number of cars available. 
             Also, the array of reward depends only on the number of requests 
             (dropoffs are here irrelevant).
             */
            satisfied_req = min(req, n); /* at most, all the cars available */
            rewards[n] += 10 * req_prob * satisfied_req ; /* +10 is the reward per request */
        }
        for (int drp = 0; (drp_prob = poisson(drp, l_drpffs)) > theta ; drp++) {
            /*
             For the calculation of the probability matrix the number of requests 
             as well as dropoffs must be considered. 
             There are different combinations of requests and dropoffs 
             that lead to the same final state s'.
             Here we sweep all the requests and dropoffs with significant probabilities 
             and sum the joint probability to the corresponding matrix element, 
             which represents a possible transition.
             */
            for (int m = 0; m < max_morning; m++) {
                satisfied_req = min(req, m);
                new_n = m + drp - satisfied_req ;
                new_n = max(new_n, 0); /* 0 at least */
                new_n = min(20, new_n); /* 20 at most */
                probs[m][new_n] += req_prob * drp_prob; /* add up the joint probability */
            }
        }
    }
}


void policy_eval()
{
    /* Policy Evaluation. */
    
    double val_tmp;
    double diff;
    int a;
    do {
        diff = 0.0;
        for (int n1 = 0; n1 < ncar_states; n1++) {
            for (int n2 = 0; n2 < ncar_states; n2++) {
                val_tmp = V[n1][n2];
                a = policy[n1][n2];
                V[n1][n2] = backup_action(n1, n2, a);
                diff = max(diff, fabs(V[n1][n2] - val_tmp));
            }
        }
    } while (diff > theta );
}


double backup_action(int n1, int n2, int a)
{
    /*
     Back up the state value of state (n1,b2) with action a.
     */
    
    double val;
    /* Determine the range of possible actions for the given state */
    a = min(a, +n1);
    a = max(a, -n2);
    a = min(+5, a);
    a = max(-5, a);
    val = -2 * fabs((double) a);
    int morning_n1 = n1 - a;
    int morning_n2 = n2 + a;
    for (int new_n1 = 0; new_n1 < ncar_states ; new_n1++) {
        for (int new_n2 = 0; new_n2 < ncar_states ; new_n2++) {
            val += prob_1[morning_n1][new_n1] * prob_2[morning_n2][new_n2] *
            (rew_1[morning_n1] + rew_2[morning_n2] + discount * V[new_n1][new_n2]);
        }
    }
    return val;
}


bool update_policy_t()
{
    /* 
     Policy Improvement.
     Return false if policy does not change.
     */
    
    int b;
    bool has_changed = false ;
    for (int n1 = 0; n1 < ncar_states; n1++) {
        for (int n2 = 0; n2 < ncar_states; n2++) {
            b = policy[n1][n2];
            policy[n1][n2] = greedy_policy(n1, n2);
            if (b != policy[n1][n2]) {
                has_changed = true;
            }
        }
    }
return (has_changed);
}


int greedy_policy(int n1, int n2)
{
    /*
     Greedily select the action that produce the highest expected reward
     after one step loop ahead of state (n1,n2).
     */
    
    /* Set the range of available actions, a belong to set A(s) */
    int a_min = max(-5,-n2);
    int a_max = min(+5,+n1);
    double val;
    double best_val;
    int best_action;
    int a;
    a = a_min;
    best_action = a_min;
    best_val = backup_action(n1, n2, a);
    for (a = a_min+1; a <= a_max; a++) {
        val = backup_action(n1, n2, a);
        if (val > best_val + 1e-9) {
            best_val = val;
            best_action = a;
        }
    }
return (best_action);
}


void print_policy()
{
    printf("\nPolicy:\n");
    for (int n1 = 0; n1 < ncar_states; n1++) {
        printf ("\n");
        for (int n2 = 0; n2 < ncar_states; n2++) {
            printf ("% 2d ", policy[ncar_states - (n1 + 1)][n2]);
        }
    }
    printf ("\n\n");
}


int main( )
{
    double lmbda_1r = 3.0; /* Request rate at location 1 */
    double lmbda_1d = 3.0; /* Drop rate at location 1 */
    double lmbda_2r = 4.0; /* Request rate at location 2 */
    double lmbda_2d = 2.0; /* Dropoff rate at location 2 */
    load_probs_rewards(prob_1, rew_1, lmbda_1r, lmbda_1d); /* 1st location */
    load_probs_rewards(prob_2, rew_2, lmbda_2r, lmbda_2d); /* 2nd location */
    
    /* 
     Policy iteration is a sequence of policy evaluation and policy improvement.
     The sequence is finished when policy improvement doesn't change the previous policy.
     */
    bool has_changed; /* Flag used to track changes in policy */
    do {
        policy_eval();
        has_changed = update_policy_t();
    } while (has_changed);
    print_policy();
return 0;
}






