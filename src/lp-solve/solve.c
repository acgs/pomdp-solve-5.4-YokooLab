#include <string.h>
#include "lpkit.h"
#include "lpglob.h"
#include "debug.h"
#include <stdio.h>
#include <gmp.h>

/* Globals used by solver */
static short JustInverted;
static short Status;
static short Doiter;
static short DoInvert;
static short Break_bb;

/* Added by A.R. Cassandra on 8/12/98 to keep a count of instability
   messages. This value is incremented */

int gLpSolveInstabilityCount = 0;
int gShowInstabilityMessages = 1;

/* Added by A. R. Cassandra on 1/11/99 to allow instability results in
   inversion to be propogated all the way out of the solution.
   Originally, inversion problems due to numerical instability would
   cause the program to exit, but I just want it to add it to the
   count of the unstable LPs. */
int gInversionProblem;

/* Added by A.R. Cassandra on 1/9/99 to keep a loop count because for
   some numerically unstable problems, the solvelp() routine's loop
   keeps going. The constant defines how many iterations to limit the
   solvelp() routine to and the count simply counts the iterations and
   terminates the loop when it reaches this level. Note that these
   cases get added into the count of unstable LPs defined in
   gLpSolveInstabilityCount. I monitored the number of iterations in a
   problem and saw a maximum of 26 and typically ones on the order of
   10.  From this I choose the loop count of 500 to be the upper limit
   and one for which we declare infeasible. If you solve problems
   where you expect the LP iterations to exceed this, then you would
   want to increase it. */
#define SOLVE_LP_MAX_LOOP_COUNT   500
unsigned int gSolveLpLoopCount;

static void ftran(lprec *lp, REAL *pcol)
{
    int  i, j, k, r, *rowp;
    REAL theta;
    REAL *valuep;
    REAL temp;

    mpq_init(*theta);
    mpq_init(*temp);

    for(i = 1; i <= lp->eta_size; i++)
    {
        k = lp->eta_col_end[i] - 1;
        r = lp->eta_row_nr[k];
        mpq_set(*theta, *pcol[r]);//theta = pcol[r];
        if(mpq_sgn(*theta) != 0)//theta != 0)
        {
            j = lp->eta_col_end[i - 1];

            /* CPU intensive loop, let's do pointer arithmetic */
            for(rowp = lp->eta_row_nr + j, valuep = lp->eta_value + j;
                j < k;
                j++, rowp++, valuep++) {
                //pcol[*rowp] += theta * *valuep;
                mpq_mul(*temp, *theta, **valuep); /*VS - Want to make sure the point arithmetic is working properly.*/
                mpq_add(*pcol[*rowp], *pcol[*rowp], *temp);
            }

            mpq_mul(*pcol[r], *pcol[r], *lp->eta_value[k]);//pcol[r] *= lp->eta_value[k];
        }
    }

    /* round small values to zero */
    /*VS - Don't need to round, since we're using rationals now */
    /*for(i = 0; i <= lp->rows; i++)
        my_round(pcol[i], lp->epsel);
    */

} /* ftran */


void lp_solve_btran(lprec *lp, REAL *row)
{
    int  i, j, k, *rowp;
    REAL f;
    REAL *valuep;
    REAL temp;
    mpq_init(*f);
    mpq_init(*temp);

    for(i = lp->eta_size; i >= 1; i--)
    {
        f = 0;
        k = lp->eta_col_end[i] - 1;
        j = lp->eta_col_end[i - 1];

        for(rowp = lp->eta_row_nr + j, valuep = lp->eta_value + j;
            j <= k;
            j++, rowp++, valuep++) {
            //f += row[*rowp] * *valuep;
            mpq_mul(*temp, *row[*rowp], **valuep);
            mpq_add(*f, *f, *temp);
        }
        /*VS - we don't need to round, since f is a rational */
        //my_round(f, lp->epsel);
        mpq_set(*row[lp->eta_row_nr[k]], *f);//row[lp->eta_row_nr[k]] = f;
    }
    mpq_clear(*f);
    mpq_clear(*temp);
} /* lp_solve_btran */


static short isvalid(lprec *lp)
{
    int i, j, *rownum, *colnum;
    int *num, row_nr;

    if(!lp->row_end_valid)
    {
        MALLOC(num, lp->rows + 1);
        MALLOC(rownum, lp->rows + 1);

        for(i = 0; i <= lp->rows; i++)
        {
            num[i] = 0;
            rownum[i] = 0;
        }

        for(i = 0; i < lp->non_zeros; i++)
            rownum[lp->mat[i].row_nr]++;

        lp->row_end[0] = 0;

        for(i = 1; i <= lp->rows; i++)
            lp->row_end[i] = lp->row_end[i - 1] + rownum[i];

        for(i = 1; i <= lp->columns; i++)
            for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
            {
                row_nr = lp->mat[j].row_nr;
                if(row_nr != 0)
                {
                    num[row_nr]++;
                    lp->col_no[lp->row_end[row_nr - 1] + num[row_nr]] = i;
                }
            }

        free(num);
        free(rownum);
        lp->row_end_valid = TRUE;
    }

    if(lp->valid)
        return(TRUE);

    CALLOC(rownum, lp->rows + 1);
    CALLOC(colnum, lp->columns + 1);

    for(i = 1 ; i <= lp->columns; i++)
        for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
        {
            colnum[i]++;
            rownum[lp->mat[j].row_nr]++;
        }

    for(i = 1; i <= lp->columns; i++)
        if(colnum[i] == 0) {
            if(lp->names_used)
                fprintf(stderr, "Warning: Variable %s not used in any constraints\n",
                        lp->col_name[i]);
            else
                fprintf(stderr, "Warning: Variable %d not used in any constraints\n",
                        i);
        }
    free(rownum);
    free(colnum);
    lp->valid = TRUE;
    return(TRUE);
}

static void resize_eta(lprec *lp)
{
    lp->eta_alloc *= 1.5;
    REALLOC(lp->eta_value, lp->eta_alloc);
    REALLOC(lp->eta_row_nr, lp->eta_alloc);
} /* resize_eta */


static void condensecol(lprec *lp,
                        int row_nr,
                        REAL *pcol)
{
    int i, elnr;

    elnr = lp->eta_col_end[lp->eta_size];

    if(elnr + lp->rows + 2 > lp->eta_alloc) /* maximum local growth of Eta */
        resize_eta(lp);

    for(i = 0; i <= lp->rows; i++)
        if(i != row_nr && pcol[i] != 0)
        {
            lp->eta_row_nr[elnr] = i;
            mpq_set(*lp->eta_value[elnr], *pcol[i]);//lp->eta_value[elnr] = pcol[i];
            elnr++;
        }

    lp->eta_row_nr[elnr] = row_nr;
    mpq_set(*lp->eta_value[elnr], *pcol[row_nr]);//lp->eta_value[elnr] = pcol[row_nr];
    elnr++;
    lp->eta_col_end[lp->eta_size + 1] = elnr;
} /* condensecol */


static void addetacol(lprec *lp)
{
    int  i, j, k;
    REAL theta;
    mpq_init(*theta);

    j = lp->eta_col_end[lp->eta_size];
    lp->eta_size++;
    k = lp->eta_col_end[lp->eta_size] - 1;
    mpq_inv(*theta, *lp->eta_value[k]);//theta = 1 / (REAL) lp->eta_value[k];
    mpq_set(*lp->eta_value[k], *theta);//lp->eta_value[k] = theta;
    //Just set theta to -theta
    mpq_neg(*theta, *theta);
    for(i = j; i < k; i++)
        mpq_mul(*lp->eta_value[i], *lp->eta_value[i], *theta);//lp->eta_value[i] *= -theta;
    JustInverted = FALSE;
} /* addetacol */


static void setpivcol(lprec *lp,
                      short lower,
                      int   varin,
                      REAL *pcol)
{
    int  i, colnr;

    for(i = 0; i <= lp->rows; i++)
        mpq_set_ui(*pcol[i], 0, 1);//pcol[i] = 0;

    if(lower)
    {
        if(varin > lp->rows)
        {
            colnr = varin - lp->rows;
            for(i = lp->col_end[colnr - 1]; i < lp->col_end[colnr]; i++)
                mpq_set(*pcol[lp->mat[i].row_nr], *lp->mat[i].value);//pcol[lp->mat[i].row_nr] = lp->mat[i].value;
            mpq_sub(*pcol[0], *pcol[0], *Extrad);//pcol[0] -= Extrad;
        }
        else
            mpq_set_ui(*pcol[varin], 1, 1);//pcol[varin] = 1;
    }
    else /* !lower */
    {
        if(varin > lp->rows)
        {
            colnr = varin - lp->rows;
            for(i = lp->col_end[colnr - 1]; i < lp->col_end[colnr]; i++)
                mpq_neg(*pcol[lp->mat[i].row_nr], *lp->mat[i].value);//pcol[lp->mat[i].row_nr] = -lp->mat[i].value;
            mpq_add(*pcol[0], *pcol[0], *Extrad);//pcol[0] += Extrad;
        }
        else
            mpq_set_si(*pcol[varin], -1, 1);//pcol[varin] = -1;
    }

    ftran(lp, pcol);
} /* setpivcol */


static void minoriteration(lprec *lp,
                           int colnr,
                           int row_nr)
{
    int  i, j, k, wk, varin, varout, elnr;
    REAL piv;
    REAL theta;
    REAL temp;
    mpq_init(*piv);
    mpq_init(*theta);
    mpq_init(*temp);

    varin = colnr + lp->rows;
    elnr = lp->eta_col_end[lp->eta_size];
    wk = elnr;
    lp->eta_size++;

    if(mpq_sgn(*Extrad) != 0)//Extrad != 0)
    {
        lp->eta_row_nr[elnr] = 0;
        mpq_neg(*lp->eta_value[elnr], *Extrad);//lp->eta_value[elnr] = -Extrad;
        elnr++;
    }

    for(j = lp->col_end[colnr - 1] ; j < lp->col_end[colnr]; j++)
    {
        k = lp->mat[j].row_nr;

        if(k == 0 && mpq_sgn(*Extrad) != 0)//Extrad != 0)
            mpq_add(*lp->eta_value[lp->eta_col_end[lp->eta_size -1]], *lp->eta_value[lp->eta_col_end[lp->eta_size -1]], *lp->mat[j].value);//lp->eta_value[lp->eta_col_end[lp->eta_size -1]] += lp->mat[j].value;
        else if(k != row_nr)
        {
            lp->eta_row_nr[elnr] = k;
            mpq_set(*lp->eta_value[elnr], *lp->mat[j].value);//lp->eta_value[elnr] = lp->mat[j].value;
            elnr++;
        }
        else
            mpq_set(*piv, *lp->mat[j].value);//piv = lp->mat[j].value;
    }

    lp->eta_row_nr[elnr] = row_nr;
    mpq_inv(*lp->eta_value[elnr], *piv);//lp->eta_value[elnr] = 1 / piv;
    mpq_div(*theta, *lp->rhs[row_nr], *piv);//theta = lp->rhs[row_nr] / piv;
    mpq_set(*lp->rhs[row_nr], *theta);//lp->rhs[row_nr] = theta;

    for(i = wk; i < elnr; i++) {
        //lp->rhs[lp->eta_row_nr[i]] -= theta * lp->eta_value[i];
        mpq_mul(*temp, *theta, *lp->eta_value[i]);
        mpq_sub(*lp->rhs[lp->eta_row_nr[i]], *lp->rhs[lp->eta_row_nr[i]], *temp);
    }

    varout = lp->bas[row_nr];
    lp->bas[row_nr] = varin;
    lp->basis[varout] = FALSE;
    lp->basis[varin] = TRUE;

    //set piv = -piv, since we use -piv throughout next loop
    mpq_neg(*piv, *piv);
    for(i = wk; i < elnr; i++)
        mpq_div(*lp->eta_value[i], *lp->eta_value[i], *piv);//lp->eta_value[i] /= -piv;

    lp->eta_col_end[lp->eta_size] = elnr + 1;

    mpq_clear(*piv);
    mpq_clear(*theta);
    mpq_clear(*temp);
} /* minoriteration */


static void rhsmincol(lprec *lp,
                      REAL theta,
                      int row_nr,
                      int varin)
{
    int  i, j, k, varout;
    REAL f;
    mpq_init(*f);

    if(row_nr > lp->rows + 1)
    {
        fprintf(stderr, "Error: rhsmincol called with row_nr: %d, rows: %d\n",
                row_nr, lp->rows);
        fprintf(stderr, "This indicates numerical instability\n");
        exit(EXIT_FAILURE);
    }

    j = lp->eta_col_end[lp->eta_size];
    k = lp->eta_col_end[lp->eta_size + 1];
    for(i = j; i < k; i++)
    {
        //f = lp->rhs[lp->eta_row_nr[i]] - theta * lp->eta_value[i];
        mpq_mul(*f, *theta, *lp->eta_value[i]);
        mpq_sub(*f, *lp->rhs[lp->eta_row_nr[i]], *f);


        /*VS - don't need to round *///my_round(f, lp->epsb);
        mpq_set(*lp->rhs[lp->eta_row_nr[i]], *f);//lp->rhs[lp->eta_row_nr[i]] = f;
    }

    mpq_set(*lp->rhs[row_nr], *theta);//lp->rhs[row_nr] = theta;
    varout = lp->bas[row_nr];
    lp->bas[row_nr] = varin;
    lp->basis[varout] = FALSE;
    lp->basis[varin] = TRUE;

    mpq_clear(*f);

} /* rhsmincol */


void invert(lprec *lp)
{
    int    i, j, v, wk, numit, varnr, row_nr, colnr, varin;
    REAL   theta;
    REAL   temp;
    REAL   *pcol;
    short  *frow;
    short  *fcol;
    int    *rownum, *col, *row;
    int    *colnum;

    mpq_init(*theta);
    mpq_init(*temp);

    /* Added 1/1//99 by A. R. Cassandra to prevent the program from
       exiting when there is an inversion problem.  Below this is now
       caught and calling points need to check for this flag. */
    gInversionProblem = FALSE;

    if(lp->print_at_invert) {
        //fprintf(stderr, "Start Invert iter %7d eta_size %4d rhs[0] %16.4f \n",
                //lp->iter, lp->eta_size, (double) - lp->rhs[0]);
        fprintf(stderr, "Start Invert iter %7d eta_size %4d rhs[0] ", lp->iter, lp->eta_size);
        mpq_neg(*temp, *lp->rhs[0]);
        mpq_out_str(stderr, 10, *temp);
    }

    CALLOC(rownum, lp->rows + 1);
    CALLOC(col, lp->rows + 1);
    CALLOC(row, lp->rows + 1);
    CALLOC(pcol, lp->rows + 1);
    CALLOC(frow, lp->rows + 1);
    CALLOC(fcol, lp->columns + 1);
    CALLOC(colnum, lp->columns + 1);

    for(i = 0; i <= lp->rows; i++)
        mpq_init(*pcol[i]);

    for(i = 0; i <= lp->rows; i++)
        frow[i] = TRUE;

    for(i = 0; i < lp->columns; i++)
        fcol[i] = FALSE;

    for(i = 0; i < lp->rows; i++)
        rownum[i] = 0;

    for(i = 0; i <= lp->columns; i++)
        colnum[i] = 0;

    for(i = 0; i <= lp->rows; i++)
        if(lp->bas[i] > lp->rows)
            fcol[lp->bas[i] - lp->rows - 1] = TRUE;
        else
            frow[lp->bas[i]] = FALSE;

    for(i = 1; i <= lp->rows; i++)
        if(frow[i])
            for(j = lp->row_end[i - 1] + 1; j <= lp->row_end[i]; j++)
            {
                wk = lp->col_no[j];
                if(fcol[wk - 1])
                {
                    colnum[wk]++;
                    rownum[i - 1]++;
                }
            }

    for(i = 1; i <= lp->rows; i++)
        lp->bas[i] = i;

    for(i = 1; i <= lp->rows; i++)
        lp->basis[i] = TRUE;

    for(i = 1; i <= lp->columns; i++)
        lp->basis[i + lp->rows] = FALSE;

    for(i = 0; i <= lp->rows; i++)
        lp->rhs[i] = lp->rh[i];

    for(i = 1; i <= lp->columns; i++)
    {
        varnr = lp->rows + i;
        if(!lp->lower[varnr])
        {
            mpq_set(*theta, *lp->upbo[varnr]);//theta = lp->upbo[varnr];
            for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
                //lp->rhs[lp->mat[j].row_nr] -= theta * lp->mat[j].value;
                mpq_mul(*temp, *theta, *lp->mat[j].value);
                mpq_sub(*lp->rhs[lp->mat[j].row_nr], *lp->rhs[lp->mat[j].row_nr], *temp);
            }
        }
    }

    for(i = 1; i <= lp->rows; i++)
        if(!lp->lower[i])
            mpq_sub(*lp->rhs[i], *lp->rhs[i], *lp->upbo[i]);//lp->rhs[i] -= lp->upbo[i];

    lp->eta_size = 0;
    v = 0;
    row_nr = 0;
    lp->num_inv = 0;
    numit = 0;

    while(v < lp->rows)
    {
        row_nr++;
        if(row_nr > lp->rows)
            row_nr = 1;

        v++;

        if(rownum[row_nr - 1] == 1)
        if(frow[row_nr])
        {
            v = 0;
            j = lp->row_end[row_nr - 1] + 1;

            while(!(fcol[lp->col_no[j] - 1]))
                j++;

            colnr = lp->col_no[j];
            fcol[colnr - 1] = FALSE;
            colnum[colnr] = 0;

            for(j = lp->col_end[colnr - 1]; j < lp->col_end[colnr]; j++)
                if(frow[lp->mat[j].row_nr])
                    rownum[lp->mat[j].row_nr - 1]--;

            frow[row_nr] = FALSE;
            minoriteration(lp, colnr, row_nr);
        }
    }
    v = 0;
    colnr = 0;
    while(v < lp->columns)
    {
        colnr++;
        if(colnr > lp->columns)
            colnr = 1;

        v++;

        if(colnum[colnr] == 1)
        if(fcol[colnr - 1])
        {
            v = 0;
            j = lp->col_end[colnr - 1] + 1;

            while(!(frow[lp->mat[j - 1].row_nr]))
                j++;

            row_nr = lp->mat[j - 1].row_nr;
            frow[row_nr] = FALSE;
            rownum[row_nr - 1] = 0;

            for(j = lp->row_end[row_nr - 1] + 1; j <= lp->row_end[row_nr]; j++)
                if(fcol[lp->col_no[j] - 1])
                    colnum[lp->col_no[j]]--;

            fcol[colnr - 1] = FALSE;
            numit++;
            col[numit - 1] = colnr;
            row[numit - 1] = row_nr;
        }
    }
    for(j = 1; j <= lp->columns; j++)
        if(fcol[j - 1])
        {
            fcol[j - 1] = FALSE;
            setpivcol(lp, lp->lower[lp->rows + j], j + lp->rows, pcol);
            row_nr = 1;

            while((row_nr <= lp->rows) && (!(frow[row_nr] && pcol[row_nr])))
                row_nr++;

            /* if(row_nr == lp->rows + 1) */

            /* Changed 1/11/99 by A. R. Cassandra to prevent the exiting of the
               program when an inversion error occurs. Old code:

               if(row_nr > lp->rows)
                   error("Inverting failed");

               Now it simply sets the global flag and there is a
               corresponding check for this flag in all the places this
               routine is called.
            */
            if(row_nr > lp->rows) {
                gInversionProblem = TRUE;
                mpq_clear(*theta);
                mpq_clear(*temp);
                return;
            } /* if inversion problem */

            frow[row_nr] = FALSE;
            condensecol(lp, row_nr, pcol);
            mpq_div(*theta, *lp->rhs[row_nr], *pcol[row_nr]);//theta = lp->rhs[row_nr] / (REAL) pcol[row_nr];
            rhsmincol(lp, theta, row_nr, lp->rows + j);
            addetacol(lp);
        }

    for(i = numit - 1; i >= 0; i--)
    {
        colnr = col[i];
        row_nr = row[i];
        varin = colnr + lp->rows;

        for(j = 0; j <= lp->rows; j++)
            mpq_set_ui(*pcol[j], 0, 1);//pcol[j] = 0;

        for(j = lp->col_end[colnr - 1]; j < lp->col_end[colnr]; j++)
            mpq_set(*pcol[lp->mat[j].row_nr], *lp->mat[j].value);//pcol[lp->mat[j].row_nr] = lp->mat[j].value;

        mpq_sub(*pcol[0], *pcol[0], *Extrad);//pcol[0] -= Extrad;
        condensecol(lp, row_nr, pcol);
        mpq_div(*theta, *lp->rhs[row_nr], *pcol[row_nr]);//theta = lp->rhs[row_nr] / (REAL) pcol[row_nr];
        rhsmincol(lp, theta, row_nr, varin);
        addetacol(lp);
    }

    /*VS - no need to round. */
    /*
    for(i = 1; i <= lp->rows; i++)
        my_round(lp->rhs[i], lp->epsb);
     */

    if(lp->print_at_invert) {
        //fprintf(stderr,
            //"End Invert                eta_size %4d rhs[0] %16.4f\n",
            //lp->eta_size, (double) - lp->rhs[0]);
        fprintf(stderr, "End Invert                eta_size %4d rhs[0] ", lp->eta_size);
        mpq_neg(*temp, *lp->rhs[0]);
        mpq_out_str(stderr, 10, *temp);
    }

    JustInverted = TRUE;
    DoInvert = FALSE;

    mpq_clear(*temp);
    mpq_clear(*theta);

    for(i = 0; i <= lp->rows; i++)
        mpq_clear(*pcol[i]);

    free(rownum);
    free(col);
    free(row);
    free(pcol);
    free(frow);
    free(fcol);
    free(colnum);
} /* invert */

static short colprim(lprec *lp,
                     int *colnr,
                     short minit,
                     REAL   *drow)
{
    int  varnr, i, j;
    REAL f;
    REAL dpiv;
    REAL temp;

    //dpiv = -lp->epsd;
    mpq_init(*dpiv);
    mpq_neg(*dpiv, *lp->epsd);

    mpq_init(*f);
    mpq_init(*temp);

    (*colnr) = 0;
    if(!minit)
    {
        for(i = 1; i <= lp->sum; i++)
            mpq_init(*drow[i]);//drow[i] = 0;
        mpq_set_ui(*drow[0], 1, 1);//drow[0] = 1;
        lp_solve_btran(lp, drow);
        for(i = 1; i <= lp->columns; i++)
        {
            varnr = lp->rows + i;
            if(!lp->basis[varnr])
            if(mpq_sgn(*lp->upbo[varnr]) > 0)//lp->upbo[varnr] > 0)
            {
                mpq_set_ui(*f, 0, 1);//f = 0;
                for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
                    //f += drow[lp->mat[j].row_nr] * lp->mat[j].value;
                    mpq_mul(*temp, *drow[lp->mat[j].row_nr], *lp->mat[j].value);
                    mpq_add(*f, *f, *temp);
                }
                mpq_set(*drow[varnr], *f);//drow[varnr] = f;
            }
        }
        /*VS - No need to round */
        //for(i = 1; i <= lp->sum; i++)
            //my_round(drow[i], lp->epsd);
    }
    for(i = 1; i <= lp->sum; i++)
        if(!lp->basis[i])
        if(mpq_sgn(*lp->upbo[i]) > 0)//lp->upbo[i] > 0)
        {
            if(lp->lower[i])
                mpq_set(*f, *drow[i]);//f = drow[i];
            else
                mpq_neg(*f, *drow[i]);//f = -drow[i];
            if(mpq_cmp(*f, *dpiv) < 0)//f < dpiv)
            {
                mpq_set(*dpiv, *f);//dpiv = f;
                (*colnr) = i;
            }
        }
    if(lp->trace) {
        if((*colnr)>0){
            //fprintf(stderr, "col_prim:%7d, reduced cost: % 18.10f\n",
                    //(*colnr), (double)dpiv);
            fprintf(stderr,"col_prim:%7d, reduced cost: ", (*colnr) );
            mpq_out_str(stderr, 10, *dpiv);
            fprintf(stderr, "\n");
        }
        else
            fprintf(stderr,
                    "col_prim: no negative reduced costs found, optimality!\n");
    }
    if(*colnr == 0)
    {
        Doiter   = FALSE;
        DoInvert = FALSE;
        Status   = OPTIMAL;
    }

    mpq_clear(*temp);
    mpq_clear(*f);
    mpq_clear(*dpiv);

    return((*colnr) > 0);
} /* colprim */

static short rowprim(lprec *lp,
                     int colnr,
                     int *row_nr,
                     REAL *theta,
                     REAL *pcol)
{
    int  i;
    //REAL f=1, quot;
    REAL f;
    REAL quot;
    REAL temp;

    mpq_init(*f);
    mpq_init(*quot);
    mpq_init(*temp);
    mpq_set_ui(*f, 1, 1);

    (*row_nr) = 0;
    mpq_set(**theta, *lp->infinite);//(*theta) = lp->infinite;
    for(i = 1; i <= lp->rows; i++)
    {
        mpq_set(*f, *pcol[i]);//f = pcol[i];
        if(mpq_sgn(*f) != 0)//f != 0)
        {
            mpq_abs(*temp, *f);
            if(mpq_cmp(*temp, *Trej) < 0)//my_abs(f) < Trej)
            {
                //debug_print(lp, "pivot %g rejected, too small (limit %g)\n",
                            //(double)f, (double)Trej);
                if(lp->debug){
                    fprintf(stderr, "pivot ");
                    mpq_out_str(stderr, 10, *f);
                    fprintf(stderr, " rejected, too small (limit ");
                    mpq_out_str(stderr, 10, *Trej);
                    fprintf(stderr, ")\n");
                }
            }
            else /* pivot alright */
            {
                /*VS- using mul_2exp, following equation becomes quot = lp->inifinie * 2^1 */ //quot = 2 * lp->infinite;
                mpq_mul_2exp(*quot, *lp->infinite, 1);
                if(mpq_sgn(*f) > 0)//f > 0)
                    mpq_div(*quot, *lp->rhs[i], *f);//quot = lp->rhs[i] / (REAL) f;
                else if(mpq_cmp(*lp->upbo[lp->bas[i]], *lp->infinite) < 0)//lp->upbo[lp->bas[i]] < lp->infinite)
                {
                    //quot = (lp->rhs[i] - lp->upbo[lp->bas[i]]) / (REAL) f;
                    mpq_sub(*temp, *lp->rhs[i], *lp->upbo[lp->bas[i]]);
                    mpq_div(*quot, *temp, *f);
                }
                /*VS - don't need to round */ //my_round(quot, lp->epsel);
                if(mpq_cmp(*quot, **theta) < 0)//quot < (*theta))
                {
                    mpq_set(**theta, *quot);//(*theta) = quot;
                    (*row_nr) = i;
                }
            }
        }
    }
    if((*row_nr) == 0)
        for(i = 1; i <= lp->rows; i++)
        {
            mpq_set(*f, *pcol[i]);//f = pcol[i];
            if(mpq_sgn(*f) != 0)//f != 0)
            {
                /*VS- using mul_2exp, following equation becomes quot = lp->inifinie * 2^1 */ //quot = 2 * lp->infinite;
                mpq_mul_2exp(*quot, *lp->infinite, 1);
                if(mpq_sgn(*f) > 0)//f > 0)
                    mpq_div(*quot, *lp->rhs[i], *f);//quot = lp->rhs[i] / (REAL) f;
                else
                    if(mpq_cmp(*lp->upbo[lp->bas[i]], *lp->infinite) < 0) {//lp->upbo[lp->bas[i]] < lp->infinite)
                        //quot = (lp->rhs[i] - lp->upbo[lp->bas[i]]) / (REAL) f;
                        mpq_sub(*temp, *lp->rhs[i], *lp->upbo[lp->bas[i]]);
                        mpq_div(*quot, *temp, *f);
                    }
                /* VS - don't need to round */ //my_round(quot, lp->epsel);
                if(mpq_cmp(*quot, **theta) < 0)//quot < (*theta))
                {
                    mpq_set(**theta, *quot);//(*theta) = quot;
                    (*row_nr) = i;
                }
            }
        }

    if(mpq_sgn(**theta) < 0)//(*theta) < 0)
    {
        /* Added by A. R. Cassandra on 8/12/98 to allow disabling and
           counting of the instability messages. */
        gLpSolveInstabilityCount++;
        if ( gShowInstabilityMessages ) {
            //fprintf(stderr, "Warning: Numerical instability, qout = %f\n",
                    //(double)(*theta));
            fprintf(stderr, "Warning: Numerical instability, qout = ");
            mpq_out_str(stderr, 10, **theta);

            //fprintf(stderr, "pcol[%d] = % 18.10f, rhs[%d] = % 18.8f , upbo = % f\n",
                    //(*row_nr), (double)f, (*row_nr), (double)lp->rhs[(*row_nr)],
                    //(double)lp->upbo[lp->bas[(*row_nr)]]);
            fprintf(stderr, "pcol[%d] = ", (*row_nr));
            mpq_out_str(stderr, 10, *f);
            fprintf(stderr, ", rhs[%d] = ", (*row_nr));
            mpq_out_str(stderr, 10, *lp->rhs[*row_nr]);
            fprintf(stderr, ", upbo = ");
            mpq_out_str(stderr, 10, *lp->upbo[lp->bas[*row_nr]]);
            fprintf(stderr, "\n");
        }
    }
    if((*row_nr) == 0)
    {
        if(mpq_equal(*lp->upbo[colnr], *lp->infinite))//lp->upbo[colnr] == lp->infinite)
        {
            Doiter   = FALSE;
            DoInvert = FALSE;
            Status   = UNBOUNDED;
        }
        else
        {
            i = 1;
            while(mpq_sgn(*pcol[i]) >= 0 && i <= lp->rows)//pcol[i] >= 0 && i <= lp->rows)
                i++;
            if(i > lp->rows) /* empty column with upperbound! */
            {
                lp->lower[colnr] = FALSE;
                //lp->rhs[0] += lp->upbo[colnr]*pcol[0];
                mpq_mul(*temp, *lp->upbo[colnr], *pcol[0]);
                mpq_add(*lp->rhs[0], *lp->rhs[0], *temp);
                Doiter = FALSE;
                DoInvert = FALSE;
            }
            else if(mpq_sgn(*pcol[i]) < 0)//pcol[i]<0)
            {
                (*row_nr) = i;
            }
        }
    }
    if((*row_nr) > 0)
        Doiter = TRUE;
    if(lp->trace) {
        //fprintf(stderr, "row_prim:%7d, pivot element:% 18.10f\n", (*row_nr),
            //(double)pcol[(*row_nr)]);
        fprintf(stderr, "row_prim:%7d, pivot element: ", *row_nr);
        mpq_out_str(stderr, 10, *pcol[*row_nr]);
        fprintf(stderr, "\n");
    }

    mpq_clear(*temp);
    mpq_clear(*f);
    mpq_clear(*quot);

    return((*row_nr) > 0);
} /* rowprim */

static short rowdual(lprec *lp, int *row_nr)
{
    int   i;
    REAL  f;
    REAL  g;
    REAL  minrhs;
    REAL  temp;
    short artifs;


    mpq_init(*f);
    mpq_init(*g);
    mpq_init(*minrhs);
    mpq_init(*temp);

    (*row_nr) = 0;
    mpq_neg(*minrhs, *lp->epsb);//minrhs = -lp->epsb;
    i = 0;
    artifs = FALSE;
    while(i < lp->rows && !artifs)
    {
        i++;
        mpq_set(*f, *lp->upbo[lp->bas[i]]);//f = lp->upbo[lp->bas[i]];
        if(mpq_sgn(*f) == 0 && mpq_sgn(*lp->rhs[i]) != 0)//f == 0 && (lp->rhs[i] != 0))
        {
            artifs = TRUE;
            (*row_nr) = i;
        }
        else
        {
            mpq_sub(*temp, *f, *lp->rhs[i]);
            if(mpq_cmp(*lp->rhs[i], *temp) < 0)//lp->rhs[i] < f - lp->rhs[i])
                mpq_set(*g, *lp->rhs[i]);//g = lp->rhs[i];
            else
                mpq_sub(*g, *f, *lp->rhs[i]);//g = f - lp->rhs[i];
            if(mpq_cmp(*g, *minrhs) < 0)//g < minrhs)
            {
                mpq_set(*minrhs, *g); //minrhs = g;
                (*row_nr) = i;
            }
        }
    }

    if(lp->trace)
    {
        if((*row_nr) > 0)
        {
            //fprintf(stderr,
                    //"row_dual:%7d, rhs of selected row:           % 18.10f\n",
                    //(*row_nr), (double)lp->rhs[(*row_nr)]);
            fprintf(stderr, "row_dual:%7d, rhs of selected row:           ", *row_nr);
            mpq_out_str(stderr, 10, *lp->rhs[*row_nr]);
            fprintf(stderr, "\n");

            if(mpq_cmp(*lp->upbo[lp->bas[*row_nr]], *lp->infinite)) {//lp->upbo[lp->bas[(*row_nr)]] < lp->infinite)
                //fprintf(stderr,
                    //"\t\tupper bound of basis variable:    % 18.10f\n",
                    //(double)lp->upbo[lp->bas[(*row_nr)]]);
                fprintf(stderr, "\t\tupper bound of basis variable:    ");
                mpq_out_str(stderr, 10, *lp->upbo[lp->bas[*row_nr]]);
                fprintf(stderr, "\n");
            }
        }
        else
            fprintf(stderr, "row_dual: no infeasibilities found\n");
    }

    mpq_clear(*temp);
    mpq_clear(*f);
    mpq_clear(*g);
    mpq_clear(*minrhs);


    return((*row_nr) > 0);
} /* rowdual */

static short coldual(lprec *lp,
                     int row_nr,
                     int *colnr,
                     short minit,
                     REAL *prow,
                     REAL *drow)
{
    int  i, j, k, r, varnr, *rowp, row;
    REAL theta;
    REAL quot;
    REAL pivot;
    REAL d;
    REAL f;
    REAL g;
    REAL *valuep;
    REAL value;
    REAL temp;
    REAL temp2;

    mpq_init(*theta);
    mpq_init(*quot);
    mpq_init(*pivot);
    mpq_init(*d);
    mpq_init(*f);
    mpq_init(*g);
    mpq_init(*value);
    mpq_init(*temp);
    mpq_init(*temp2);

    Doiter = FALSE;
    if(!minit)
    {
        for(i = 0; i <= lp->rows; i++)
        {
            mpq_init(*prow[i]);//prow[i] = 0;
            mpq_init(*drow[i]);//drow[i] = 0;
        }

        mpq_set_ui(*drow[0], 1, 1);//drow[0] = 1;
        mpq_set_ui(*prow[row_nr], 1, 1);//prow[row_nr] = 1;

        for(i = lp->eta_size; i >= 1; i--)
        {
            mpq_set_ui(*d, 0, 1);//d = 0;
            mpq_set_ui(*f, 0, 1);//f = 0;
            k = lp->eta_col_end[i] - 1;
            r = lp->eta_row_nr[k];
            j = lp->eta_col_end[i - 1];

            /* VS - compilers have come a long way - maybe we should just rewrite this to use array indexing for readability */
            /* this is one of the loops where the program consumes a lot of CPU
                   time */
            /* let's help the compiler by doing some pointer arithmetic instead
                   of array indexing */
            for(rowp = lp->eta_row_nr + j, valuep = lp->eta_value + j;
                j <= k;
                j++, rowp++, valuep++)
            {
                //f += prow[*rowp] * *valuep;
                mpq_mul(*temp, *prow[*rowp], **valuep);
                mpq_add(*f, *f, *temp);

                //d += drow[*rowp] * *valuep;
                mpq_mul(*temp, *drow[*rowp], **valuep);
                mpq_add(*d, *d, *temp);
            }

            /*VS - don't need to round */ //my_round(f, lp->epsel);
            mpq_set(*prow[r], *f); //prow[r] = f;
            /*VS - don't need to round */ //my_round(d, lp->epsd);
            mpq_set(*drow[r], *d); //drow[r] = d;
        }

        for(i = 1; i <= lp->columns; i++)
        {
            varnr = lp->rows + i;
            if(!lp->basis[varnr])
            {
                matrec *matentry;

                //d = - Extrad * drow[0];
                mpq_neg(*temp, *Extrad);
                mpq_mul(*d, *temp, *drow[0]);

                mpq_set_ui(*f, 0, 1);//f = 0;

                k = lp->col_end[i];
                j = lp->col_end[i - 1];

                /* VS - compilers have come a long way - maybe we should just rewrite this to use array indexing for readability */
                /* this is one of the loops where the program consumes a lot
               of cpu time */
                /* let's help the compiler with pointer arithmetic instead
               of array indexing */
                for(matentry = lp->mat + j;
                    j < k;
                    j++, matentry++)
                {
                    row = (*matentry).row_nr;
                    mpq_set(*value, *(*matentry).value);//value = (*matentry).value;
                    //d += drow[row] * value;
                    mpq_mul(*temp, *drow[row], *value);
                    mpq_add(*d, *d, *temp);

                    //f += prow[row] * value;
                    mpq_mul(*temp, *prow[row], *value);
                    mpq_add(*f, *f, *temp);
                }

                /*VS - don't need to round */ //my_round(f, lp->epsel);
                mpq_set(*prow[varnr], *f); //prow[varnr] = f;
                /*VS - don't need to round */ //my_round(d, lp->epsd);
                mpq_set(*drow[varnr], *d); //drow[varnr] = d;
            }
        }
    }

    if(mpq_cmp(*lp->rhs[row_nr], *lp->upbo[lp->bas[row_nr]]) > 0)//lp->rhs[row_nr] > lp->upbo[lp->bas[row_nr]])
        mpq_set_si(*g, -1, 1); //g = -1;
    else
        mpq_set_ui(*g, 1, 1); //g = 1;

    pivot = 0;
    (*colnr) = 0;
    mpq_set(*theta, *lp->infinite);//theta = lp->infinite;

    for(i = 1; i <= lp->sum; i++)
    {
        if(lp->lower[i])
            mpq_mul(*d, *prow[i], *g); //d = prow[i] * g;
        else {
            mpq_neg(*temp, *prow[i]); //d = -prow[i] * g;
            mpq_mul(*d, *temp, *g);
        }


        if(mpq_sgn(*d) < 0 && !lp->basis[i] && mpq_sgn(*lp->upbo[i]) > 0)//(d < 0) && (!lp->basis[i]) && (lp->upbo[i] > 0))
        {
            if(lp->lower[i]) {
                //quot = -drow[i] / (REAL) d;
                mpq_neg(*temp, *drow[i]);
                mpq_div(*quot, *temp, *d);
            }
            else
                //quot = drow[i] / (REAL) d;
                mpq_div(*quot, *drow[i], *d);

            mpq_abs(*temp, *d);
            mpq_abs(*temp2, *pivot);
            if(mpq_cmp(*quot, *theta) < 0)//quot < theta)
            {
                mpq_set(*theta, *quot); //theta = quot;
                mpq_set(*pivot, *d); //pivot = d;
                (*colnr) = i;
            }
            else if(mpq_cmp(*quot, *theta) == 0 && mpq_cmp(*temp, *temp2) > 0)//(quot == theta) && (my_abs(d) > my_abs(pivot)))
            {
                mpq_set(*pivot, *d); //pivot = d;
                (*colnr) = i;
            }
        }
    }

    if(lp->trace) {
        //fprintf(stderr, "col_dual:%7d, pivot element:  % 18.10f\n", (*colnr),
                //(double) prow[(*colnr)]);
        fprintf(stderr, "col_dual:%7d, pivot element:  ", (*colnr));
        mpq_out_str(stderr, 10, *prow[*colnr]);
        fprintf(stderr, "\n");
    }

    if((*colnr) > 0)
        Doiter = TRUE;


    mpq_clear(*theta);
    mpq_clear(*quot);
    mpq_clear(*pivot);
    mpq_clear(*d);
    mpq_clear(*f);
    mpq_clear(*g);
    mpq_clear(*value);
    mpq_clear(*temp);
    mpq_clear(*temp2);


    return((*colnr) > 0);
} /* coldual */

static void iteration(lprec *lp,
                      int row_nr,
                      int varin,
                      REAL *theta,
                      REAL up,
                      short *minit,
                      short *low,
                      short primal,
                      REAL *pcol)
{
    int  i, k, varout;
    REAL f;
    REAL pivot;
    REAL temp;


    mpq_init(*f);
    mpq_init(*pivot);
    mpq_init(*temp);

    lp->iter++;

    /* VS - mpq_cmp evaluates differently than >, so we need to rework the original comparison/assignment below */

    /*if(((*minit) = (*theta) > (up + lp->epsb)))
    {
        (*theta) = up;
        (*low) = !(*low);
    }
    */
    mpq_add(*temp, *up, *lp->epsb);
    if(mpq_cmp(**theta, *temp)){
        //VS - we set minit to 1 here, since the comparison returned true
        *minit = 1;
        mpq_set(**theta, *up);
        *low = !(*low);
    }
    else{
        *minit = 0;
    }

    k = lp->eta_col_end[lp->eta_size + 1];
    mpq_set(*pivot, *lp->eta_value[k-1]);//pivot = lp->eta_value[k - 1];

    for(i = lp->eta_col_end[lp->eta_size]; i < k; i++)
    {
        //f = lp->rhs[lp->eta_row_nr[i]] - (*theta) * lp->eta_value[i];
        mpq_mul(*temp, **theta, *lp->eta_value[i]);
        mpq_sub(*f, *lp->rhs[lp->eta_row_nr[i]], *temp);

        /*VS - don't need to round *///my_round(f, lp->epsb);
        mpq_set(*lp->rhs[lp->eta_row_nr[i]], *f); //lp->rhs[lp->eta_row_nr[i]] = f;
    }

    if(!(*minit))
    {
        mpq_set(*lp->rhs[row_nr], **theta);//lp->rhs[row_nr] = (*theta);
        varout = lp->bas[row_nr];
        lp->bas[row_nr] = varin;
        lp->basis[varout] = FALSE;
        lp->basis[varin] = TRUE;

        if(primal && pivot < 0)
            lp->lower[varout] = FALSE;

        if(!(*low) && mpq_cmp(*up, *lp->infinite) < 0)//&& up < lp->infinite)
        {
            (*low) = TRUE;
            mpq_sub(*lp->rhs[row_nr], *up, *lp->rhs[row_nr]);//lp->rhs[row_nr] = up - lp->rhs[row_nr];
            for(i = lp->eta_col_end[lp->eta_size]; i < k; i++)
                mpq_neg(*lp->eta_value[i], *lp->eta_value[i]);//lp->eta_value[i] = -lp->eta_value[i];
        }

        addetacol(lp);
        lp->num_inv++;
    }

    if(lp->trace)
    {
        //fprintf(stderr, "Theta = %16.4g ", (double)(*theta));
        fprintf(stderr, "Theta = ");
        mpq_out_str(stderr, 10, **theta);
        fprintf(stderr, " ");
        if((*minit))
        {
            if(!lp->lower[varin]) {
                //fprintf(stderr,
                        //"Iteration:%6d, variable%5d changed from 0 to its upper bound of %12f\n",
                        //lp->iter, varin, (double) lp->upbo[varin]);
                fprintf(stderr,"Iteration:%6d, variable%5d changed from 0 to its upper bound of ", lp->iter, varin );
                mpq_out_str(stderr, 10, *lp->upbo[varin]);
                fprintf(stderr, "\n");
            }
            else {
                //fprintf(stderr,
                    //"Iteration:%6d, variable%5d changed its upper bound of %12f to 0\n",
                    //lp->iter, varin, (double)lp->upbo[varin]);
                fprintf(stderr, "Iteration:%6d, variable%5d changed its upper bound of ", lp->iter, varin);
                mpq_out_str(stderr, 10, *lp->upbo[varin]);
                fprintf(stderr, " to 0\n");
            }
        }
        else {
            //fprintf(stderr,
                    //"Iteration:%6d, variable%5d entered basis at:% 18.10f\n",
                    //lp->iter, varin, (double) lp->rhs[row_nr]);
            fprintf(stderr, "Iteration:%6d, variable%5d entered basis at: ",lp->iter, varin);
            mpq_out_str(stderr, 10, *lp->rhs[row_nr]);
            fprintf(stderr, "\n");
        }
        if(!primal)
        {
            mpq_set_ui(*f, 0, 1);//f = 0;
            for(i = 1; i <= lp->rows; i++)
                if(mpq_sgn(*lp->rhs[i]) < 0)//lp->rhs[i] < 0)
                    mpq_sub(*f, *f, *lp->rhs[i]);//f -= lp->rhs[i];
                else
                if(mpq_cmp(*lp->rhs[i], *lp->upbo[lp->bas[i]]) > 0)//lp->rhs[i] > lp->upbo[lp->bas[i]])
                {
                    //f += lp->rhs[i] - lp->upbo[lp->bas[i]];
                    mpq_sub(*temp, *lp->rhs[i], *lp->upbo[lp->bas[i]]);
                    mpq_add(*f, *f, *temp);
                }
            //fprintf(stderr, "feasibility gap of this basis:% 18.10f\n",
                    //(double)f);
            fprintf(stderr, "feasibility gap of this basis: ");
            mpq_out_str(stderr, 10, *f);
            fprintf(stderr, "\n");
        }
        else {
            //fprintf(stderr,
                //"objective function value of this feasible basis: % 18.10f\n",
                //(double)lp->rhs[0]);
            fprintf(stderr, "objective function value of this feasible basis: ");
            mpq_out_str(stderr, 10, *lp->rhs[0]);
            fprintf(stderr, "\n");
        }
    }
    mpq_clear(*f);
    mpq_clear(*temp);
    mpq_clear(*pivot);
} /* iteration */


static int solvelp(lprec *lp)
{
    int    i, j, varnr;
    REAL   f;
    REAL   theta;
    short  primal;
    REAL   *drow;
    REAL   *prow;
    REAL   *Pcol;
    short  minit;
    int    colnr, row_nr;
    short  *test;
    REAL   temp;

    if(lp->do_presolve)
        lp_solve_presolve(lp);

    CALLOC(drow, lp->sum + 1);
    CALLOC(prow, lp->sum + 1);
    CALLOC(Pcol, lp->rows + 1);
    CALLOC(test, lp->sum +1);

    //VS - initialize drow, prow, and Pcol
    for(i = 0; i <= lp->sum; i++){
        mpq_init(*drow[i]);
        mpq_init(*prow[i]);
    }
    for(i = 0; i <= lp->rows; i++){
        mpq_init(*Pcol[i]);
    }

    mpq_init(*temp);

    lp->iter = 0;
    minit = FALSE;
    Status = RUNNING;
    DoInvert = FALSE;
    Doiter = FALSE;

    for(i = 1, primal = TRUE; (i <= lp->rows) && primal; i++)
        primal = (mpq_sgn(*lp->rhs[i]) >= 0) && (mpq_cmp(*lp->rhs[i], *lp->upbo[lp->bas[i]]) <= 0);//(lp->rhs[i] >= 0) && (lp->rhs[i] <= lp->upbo[lp->bas[i]]);

    if(lp->trace)
    {
        if(primal)
            fprintf(stderr, "Start at feasible basis\n");
        else
            fprintf(stderr, "Start at infeasible basis\n");
    }

    if(!primal)
    {
        mpq_set_ui(*drow[0], 1, 1);//drow[0] = 1;

        /* VS - don't need this loop, since drow is already initialized to 0 above */
        //for(i = 1; i <= lp->rows; i++)
            //drow[i] = 0;

        mpq_set_ui(*Extrad, 0, 1);//Extrad = 0;

        for(i = 1; i <= lp->columns; i++)
        {
            varnr = lp->rows + i;
            mpq_set_ui(*drow[varnr], 0, 1); //drow[varnr] = 0;

            for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
                if(mpq_sgn(*drow[lp->mat[j].row_nr]) != 0)//drow[lp->mat[j].row_nr] != 0)
                {
                    //drow[varnr] += drow[lp->mat[j].row_nr] * lp->mat[j].value;
                    mpq_mul(*temp, *drow[lp->mat[j].row_nr], *lp->mat[j].value);
                    mpq_add(*drow[varnr], *drow[varnr], *temp);
                }

            if(mpq_cmp(*drow[varnr], *Extrad) < 0)//drow[varnr] < Extrad)
                mpq_set(*Extrad, *drow[varnr]);//Extrad = drow[varnr];
        }
    }
    else
        mpq_set_ui(*Extrad, 0, 1);//Extrad = 0;

    if(lp->trace) {
        //fprintf(stderr, "Extrad = %f\n", (double) Extrad);
        fprintf(stderr, "Extrad = ");
        mpq_out_str(stderr, 10, *Extrad);
        fprintf(stderr, "\n");
    }

    minit = FALSE;

    /* Added 1/9/99 by arc:

       This initialization line and the first line in the loop
       testing and incrementing it to detect LPs that loop
       indefinitely. */
    gSolveLpLoopCount = 0;
    while(Status == RUNNING)
    {
        /* Added 1/9/99 by arc */
        gSolveLpLoopCount++;
        if ( gSolveLpLoopCount == SOLVE_LP_MAX_LOOP_COUNT ) {

            gLpSolveInstabilityCount++;
            if ( gShowInstabilityMessages )
                fprintf( stderr,
                         "*** solvelp() loop count reached %d ***\n",
                         SOLVE_LP_MAX_LOOP_COUNT );

            Status = FAILURE;
            break;
        } /* If maximum loop count reached */

        Doiter = FALSE;
        DoInvert = FALSE;

        if(primal)
        {
            if(colprim(lp, &colnr, minit, drow))
            {
                setpivcol(lp, lp->lower[colnr], colnr, Pcol);

                if(rowprim(lp, colnr, &row_nr, &theta, Pcol))
                    condensecol(lp, row_nr, Pcol);
            }
        }
        else /* not primal */
        {
            if(!minit)
                rowdual(lp, &row_nr);

            if(row_nr > 0 )
            {
                if(coldual(lp, row_nr, &colnr, minit, prow, drow))
                {
                    setpivcol(lp, lp->lower[colnr], colnr, Pcol);

                    /* getting div by zero here. Catch it and try to recover */
                    if(mpq_sgn(*Pcol[row_nr]) == 0)//Pcol[row_nr] == 0)
                    {

                        /* Added 1/11/99 by A.R. Cassandra. Previously this
                           used to just print out a message, but now I tie
                           it to the gShowInstabilityMessages flag and
                           increment the instability count if more serious
                           problems occur. It used to try to recover, but
                           now I just mark it as failure.

                           Old code:

                           Doiter = FALSE;
                           if(!JustInverted)
                             {
                               fprintf(stderr,
                       "Trying to recover. Reinverting Eta\n");
                               DoInvert = TRUE;
                             }
                           else
                             {
                               fprintf(stderr, "Can't reinvert, failure\n");
                               Status = FAILURE;
                             }
                        */
                        if ( gShowInstabilityMessages ) {
                            fprintf(stderr,
                                    "An attempt was made to divide by zero (Pcol[%d])\n",
                                    row_nr);
                            fprintf(stderr,
                                    "This indicates numerical instability\n");
                        }

                        gLpSolveInstabilityCount++;
                        Doiter = FALSE;
                        Status = FAILURE;
                    }
                    else
                    {
                        condensecol(lp, row_nr, Pcol);
                        mpq_sub(*f, *lp->rhs[row_nr], *lp->upbo[lp->bas[row_nr]]);//f = lp->rhs[row_nr] - lp->upbo[lp->bas[row_nr]];

                        if(mpq_sgn(*f) > 0)//f > 0)
                        {
                            mpq_div(*theta, *f, *Pcol[row_nr]);//theta = f / (REAL) Pcol[row_nr];
                            if(mpq_cmp(*theta, *lp->upbo[colnr]) <= 0)//theta <= lp->upbo[colnr])
                                lp->lower[lp->bas[row_nr]] =
                                        !lp->lower[lp->bas[row_nr]];
                        }
                        else /* f <= 0 */
                            mpq_div(*theta, *lp->rhs[row_nr], *Pcol[row_nr]); //theta = lp->rhs[row_nr] / (REAL) Pcol[row_nr];
                    }
                }
                else
                    Status = INFEASIBLE;
            }
            else
            {
                primal   = TRUE;
                Doiter   = FALSE;
                Extrad   = 0;
                DoInvert = TRUE;
            }
        }

        if(Doiter)
            iteration(lp, row_nr, colnr, &theta, lp->upbo[colnr], &minit,
                      &lp->lower[colnr], primal, Pcol);

        if(lp->num_inv >= lp->max_num_inv)
            DoInvert = TRUE;

        if(DoInvert)
        {
            if(lp->print_at_invert)
                fprintf(stderr, "Inverting: Primal = %d\n", primal);
            invert(lp);

            /* Added 1/11/99 to catch whether or not there was an
               inversion problem. We assume that an inversion problem is
               due to numerical stability issues. */
            if ( gInversionProblem ) {

                gLpSolveInstabilityCount++;
                Status = FAILURE;

                if ( gShowInstabilityMessages ) {
                    fprintf(stderr, "Warning: Inversion problem." );
                    fprintf(stderr, "Numerical instability?\n" );
                }
            } /* if inversion problem */

        }
    }

    lp->total_iter += lp->iter;

    for(i = 0; i <= lp->sum; i++){
        mpq_clear(*drow[i]);
        mpq_clear(*prow[i]);
    }
    for(i = 0; i <= lp->rows; i++){
        mpq_clear(*Pcol[i]);
    }
    mpq_clear(*temp);
    mpq_clear(*f);
    mpq_clear(*theta);

    free(drow);
    free(prow);
    free(Pcol);
    free(test);

    return(Status);
} /* solvelp */


static short is_int(lprec *lp, REAL value)
{

    /*VS - we can replace all of this with a call to canonicalize on value and check if denominator is 1 */
    mpq_canonicalize(*value);
    return mpz_cmp_ui(mpq_denref(*value), 1);
    /*
    REAL   tmp;

    tmp = value - (REAL)floor((double)value);

    if(tmp < lp->epsilon)
        return(TRUE);

    if(tmp > (1 - lp->epsilon))
        return(TRUE);

    return(FALSE);
    */
} /* is_int */


static void construct_solution(lprec *lp)
{
    int    i, j, basi;
    REAL   f;
    REAL   temp;
    REAL   temp2;

    mpq_init(*f);
    mpq_init(*temp);
    mpq_init(*temp2);

    /* zero all results of rows */
    memset(lp->solution, '\0', (lp->rows + 1) * sizeof( mpq_t));

    mpq_neg(*lp->solution[0], *lp->orig_rh[0]);//lp->solution[0] = -lp->orig_rh[0];

    if(lp->scaling_used)
    {
        mpq_div(*lp->solution[0], *lp->solution[0], *lp->scale[0]);//lp->solution[0] /= lp->scale[0];

        for(i = lp->rows + 1; i <= lp->sum; i++)
            mpq_mul(*lp->solution[i], *lp->lowbo[i], *lp->scale[i]);//lp->solution[i] = lp->lowbo[i] * lp->scale[i];
        for(i = 1; i <= lp->rows; i++)
        {
            basi = lp->bas[i];
            if(basi > lp->rows) {
                //lp->solution[basi] += lp->rhs[i] * lp->scale[basi];
                mpq_mul(*temp, *lp->rhs[i], *lp->scale[basi]);
                mpq_add(*lp->solution[basi], *lp->solution[basi], *temp);
            }
        }
        for(i = lp->rows + 1; i <= lp->sum; i++)
            if(!lp->basis[i] && !lp->lower[i]) {
                //lp->solution[i] += lp->upbo[i] * lp->scale[i];
                mpq_mul(*temp, *lp->upbo[i], *lp->scale[i]);
                mpq_add(*lp->solution[i], *lp->solution[i], *temp);
            }

        for(j = 1; j <= lp->columns; j++)
        {
            mpq_set(*f, *lp->solution[lp->rows + j]);//f = lp->solution[lp->rows + j];
            if(mpq_sgn(*f) != 0)//f != 0)
                for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++) {
                    //lp->solution[lp->mat[i].row_nr] += (f / lp->scale[lp->rows + j])
                                                       //* (lp->mat[i].value / lp->scale[lp->mat[i].row_nr]);
                    mpq_div(*temp, *f, *lp->scale[lp->rows + j]);
                    mpq_div(*temp2, *lp->mat[i].value, *lp->scale[lp->mat[i].row_nr]);
                    mpq_mul(*temp, *temp, *temp2);
                    mpq_add(*lp->solution[lp->mat[i].row_nr], *lp->solution[lp->mat[i].row_nr], *temp);
                }
        }

        for(i = 0; i <= lp->rows; i++)
        {
            mpq_abs(*temp, *lp->solution[i]);
            if(mpq_cmp(*temp, *lp->epsb) < 0)//my_abs(lp->solution[i]) < lp->epsb)
                mpq_set_ui(*lp->solution[i], 0, 1);//lp->solution[i] = 0;
            else if(lp->ch_sign[i])
                mpq_neg(*lp->solution[i], *lp->solution[i]);//lp->solution[i] = -lp->solution[i];
        }
    }
    else /* no scaling */
    {
        for(i = lp->rows + 1; i <= lp->sum; i++)
            mpq_set(*lp->solution[i], *lp->lowbo[i]);//lp->solution[i] = lp->lowbo[i];
        for(i = 1; i <= lp->rows; i++)
        {
            basi = lp->bas[i];
            if(basi > lp->rows)
                mpq_add(*lp->solution[basi], *lp->solution[basi], *lp->rhs[i]);//lp->solution[basi] += lp->rhs[i];
        }
        for(i = lp->rows + 1; i <= lp->sum; i++)
            if(!lp->basis[i] && !lp->lower[i])
                mpq_add(*lp->solution[i], *lp->solution[i], *lp->upbo[i]);//lp->solution[i] += lp->upbo[i];
        for(j = 1; j <= lp->columns; j++)
        {
            mpq_set(*f, *lp->solution[lp->rows + j]);//f = lp->solution[lp->rows + j];
            if(mpq_sgn(*f) != 0)//f != 0)
                for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++) {
                    //lp->solution[lp->mat[i].row_nr] += f * lp->mat[i].value;
                    mpq_mul(*temp, *f, *lp->mat[i].value);
                    mpq_add(*lp->solution[lp->mat[i].row_nr], *lp->solution[lp->mat[i].row_nr], *temp);
                }
        }

        for(i = 0; i <= lp->rows; i++)
        {
            /* VS - maybe we don't need to round down */

            if(lp->ch_sign[i])
                mpq_neg(*lp->solution[i], *lp->solution[i]);

            /*
            if(my_abs(lp->solution[i]) < lp->epsb)
                lp->solution[i] = 0;
            else if(lp->ch_sign[i])
                lp->solution[i] = -lp->solution[i];
            */
        }
    }

    mpq_clear(*temp);
    mpq_clear(*f);
    mpq_clear(*temp2);

} /* construct_solution */

static void calculate_duals(lprec *lp)
{
    int i;
    REAL temp;
    mpq_init(*temp);

    /* initialize */
    /* VS - We assume lp->duals has had its memory allocated elsewhere*/
    //lp->duals[0] = 1;
    mpq_set_ui(*lp->duals[0], 1, 1);
    for(i = 1; i <= lp->rows; i++)
        mpq_set_ui(*lp->duals[i], 0, 1); //lp->duals[i] = 0;

    lp_solve_btran(lp, lp->duals);

    if(lp->scaling_used)
        for(i = 1; i <= lp->rows; i++) {
            //lp->duals[i] *= lp->scale[i] / lp->scale[0];
            mpq_div(*temp, *lp->scale[i], *lp->scale[0]);
            mpq_mul(*lp->duals[i], *lp->duals[i], *temp);
        }

    /* the dual values are the reduced costs of the slacks */
    /* When the slack is at its upper bound, change the sign. */
    for(i = 1; i <= lp->rows; i++)
    {
        if(lp->basis[i])
            mpq_set_ui(*lp->duals[i], 0, 1);//lp->duals[i] = 0;
            /* added a test if variable is different from 0 because sometime you get
               -0 and this is different from 0 on for example INTEL processors (ie 0
               != -0 on INTEL !) PN */
        else if((lp->ch_sign[0] == lp->ch_sign[i]) && mpq_sgn(*lp->duals[i]) != 0)//lp->duals[i])
            mpq_neg(*lp->duals[i], *lp->duals[i]);//lp->duals[i] = - lp->duals[i];
    }
} /* calculate_duals */


/*VS - not sure what the purpose of this function is besides printing an error message - maybe we should just
 * change the function name to be more meaningful, since it doesn't actually return anything. */
static void check_if_less(REAL x,
                          REAL y,
                          REAL value)
{
    if(x >= y)
    {
        fprintf(stderr,
                "Error: new upper or lower bound is not more restrictive\n");
        //fprintf(stderr, "bound 1: %g, bound 2: %g, value: %g\n",
                //(double)x, (double)y, (double)value);
        fprintf(stderr, "bound 1: " );
        mpq_out_str(stderr, 10, *x);
        fprintf(stderr, ", bound 2: ");
        mpq_out_str(stderr, 10, *y);
        fprintf(stderr, ", value: ");
        mpq_out_str(stderr, 10, *value);
        fprintf(stderr, "\n");
        /* exit(EXIT_FAILURE); */
    }
}


/* This function is never called.  There is a commented out call later
   in the file around line 1454, but to prevent a compiler warning I
   use conditional compilation to make it ignore this. This was done
   on 1/9/99 by A. R. Cassandra. */
#ifdef DEFINE_CHECK_SOLUTION
static void check_solution(lprec *lp,
               REAL *upbo,
               REAL *lowbo)
{
  int i;

  /* check if all solution values are within the bounds, but allow some margin
     for numerical errors */

#define CHECK_EPS 1e-2

  if(lp->columns_scaled)
    for(i = lp->rows + 1; i <= lp->sum; i++)
      {
    if(lp->solution[i] < lowbo[i] * lp->scale[i] - CHECK_EPS)
      {
        fprintf(stderr,
            "Error: variable %d (%s) has a solution (%g) smaller than its lower bound (%g)\n",
            i - lp->rows, lp->col_name[i - lp->rows], lp->solution[i],
            lowbo[i] * lp->scale[i]);
        /* abort(); */
      }

    if(lp->solution[i] > upbo[i] * lp->scale[i] + CHECK_EPS)
      {
        fprintf(stderr,
            "Error: variable %d (%s) has a solution (%g) larger than its upper bound (%g)\n",
            i - lp->rows, lp->col_name[i - lp->rows], lp->solution[i],
            upbo[i] * lp->scale[i]);
        /* abort(); */
      }
      }
  else /* columns not scaled */
    for(i = lp->rows + 1; i <= lp->sum; i++)
      {
    if(lp->solution[i] < lowbo[i] - CHECK_EPS)
      {
        fprintf(stderr,
            "Error: variable %d (%s) has a solution (%g) smaller than its lower bound (%g)\n",
            i - lp->rows, lp->col_name[i - lp->rows], lp->solution[i], lowbo[i]);
        /* abort(); */
      }

    if(lp->solution[i] > upbo[i] + CHECK_EPS)
      {
        fprintf(stderr,
            "Error: variable %d (%s) has a solution (%g) larger than its upper bound (%g)\n",
            i - lp->rows, lp->col_name[i - lp->rows], lp->solution[i], upbo[i]);
        /* abort(); */
      }
      }
} /* check_solution */
#endif
/* End of conditional compilation of this function. */

static int milpsolve(lprec *lp,
                     REAL   *upbo,
                     REAL   *lowbo,
                     short  *sbasis,
                     short  *slower,
                     int    *sbas,
                     int     recursive)
{
    int i, j, failure, notint=0, is_worse;
    REAL theta;
    REAL tmpreal;
    REAL temp;
    mpz_t tmpInt;

    mpq_init(*theta);
    mpq_init(*tmpreal);
    mpq_init(*temp);
    mpz_init(tmpInt);
    mpz_set_ui(tmpInt, 100);

    if(Break_bb) {
        mpq_clear(*theta);
        mpq_clear(*tmpreal);
        mpq_clear(*temp);
        mpz_clear(tmpInt);
        return (BREAK_BB);
    }

    Level++;
    lp->total_nodes++;

    if(Level > lp->max_level)
        lp->max_level = Level;

    debug_print(lp, "starting solve");

    /* make fresh copies of upbo, lowbo, rh as solving changes them */
    memcpy(lp->upbo,  upbo,    (lp->sum + 1)  * sizeof(mpq_t));
    memcpy(lp->lowbo, lowbo,   (lp->sum + 1)  * sizeof(mpq_t));
    memcpy(lp->rh,    lp->orig_rh, (lp->rows + 1) * sizeof(mpq_t));

    /* make shure we do not do memcpy(lp->basis, lp->basis ...) ! */
    if(recursive)
    {
        memcpy(lp->basis, sbasis,  (lp->sum + 1)  * sizeof(short));
        memcpy(lp->lower, slower,  (lp->sum + 1)  * sizeof(short));
        memcpy(lp->bas,   sbas,    (lp->rows + 1) * sizeof(int));
    }

    if(lp->anti_degen) /* randomly disturb bounds */
    {
        mpq_set_d(*temp, 0.00001);
        for(i = 1; i <= lp->columns; i++)
        {
            //tmpreal = (REAL) (rand() % 100 * 0.00001);
            mpz_random(mpq_numref(*tmpreal), 100);
            mpz_random(mpq_denref(*tmpreal), 100);
            mpz_mod(mpq_numref(*tmpreal), mpq_numref(*tmpreal), tmpInt);
            mpz_mod(mpq_denref(*tmpreal), mpq_denref(*tmpreal), tmpInt);
            mpq_mul(*tmpreal, *tmpreal, *temp);

            if(mpq_cmp(*tmpreal, *lp->epsb) > 0)//tmpreal > lp->epsb)
                mpq_sub(*lp->lowbo[i + lp->rows], *lp->lowbo[i + lp->rows], *tmpreal);//lp->lowbo[i + lp->rows] -= tmpreal;
            //tmpreal = (REAL) (rand() % 100 * 0.00001);
            mpz_random(mpq_numref(*tmpreal), 100);
            mpz_random(mpq_denref(*tmpreal), 100);
            mpz_mod(mpq_numref(*tmpreal), mpq_numref(*tmpreal), tmpInt);
            mpz_mod(mpq_denref(*tmpreal), mpq_denref(*tmpreal), tmpInt);
            mpq_mul(*tmpreal, *tmpreal, *temp);
            if(mpq_cmp(*tmpreal, *lp->epsb) > 0)//tmpreal > lp->epsb)
                mpq_add(*lp->lowbo[i + lp->rows], *lp->lowbo[i + lp->rows], *tmpreal);//lp->lowbo[i + lp->rows] += tmpreal;
        }
        lp->eta_valid = FALSE;
    }

    if(!lp->eta_valid)
    {
        /* transform to all lower bounds to zero */
        for(i = 1; i <= lp->columns; i++)
            mpq_set(*theta, *lp->lowbo[lp->rows + i]);
            if(mpq_sgn(*theta) != 0)//if((theta = lp->lowbo[lp->rows + i]) != 0)
            {
                if(mpq_cmp(*lp->upbo[lp->rows + i], *lp->infinite) < 0) //lp->upbo[lp->rows + i] < lp->infinite)
                    mpq_sub(*lp->upbo[lp->rows + i], *lp->upbo[lp->rows + i], *theta); //lp->upbo[lp->rows + i] -= theta;
                for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
                    //lp->rh[lp->mat[j].row_nr] -= theta * lp->mat[j].value;
                    mpq_mul(*temp, *theta, *lp->mat[j].value);
                    mpq_sub(*lp->rh[lp->mat[j].row_nr], *lp->rh[lp->mat[j].row_nr], *temp);
                }
            }
        invert(lp);
        lp->eta_valid = TRUE;
    }

    failure = solvelp(lp);

    if(lp->anti_degen)
        /* restore to original problem, solve again starting from the basis found
           for the disturbed problem */
    {
        /* restore original problem */
        memcpy(lp->upbo,  upbo,        (lp->sum + 1)  * sizeof(mpq_t));
        memcpy(lp->lowbo, lowbo,       (lp->sum + 1)  * sizeof(mpq_t));
        memcpy(lp->rh,    lp->orig_rh, (lp->rows + 1) * sizeof(mpq_t));

        /* transform to all lower bounds zero */
        for(i = 1; i <= lp->columns; i++) {
            mpq_set(*theta, *lp->lowbo[lp->rows + i]);
            if(mpq_sgn(*theta) != 0){//if ((theta = lp->lowbo[lp->rows + i] != 0)) {
                if (mpq_cmp(*lp->upbo[lp->rows + i], *lp->infinite))//lp->upbo[lp->rows + i] < lp->infinite)
                    mpq_sub(*lp->upbo[lp->rows + i], *lp->upbo[lp->rows + i], *theta);//lp->upbo[lp->rows + i] -= theta;
                for (j = lp->col_end[i - 1]; j < lp->col_end[i]; j++) {
                    //lp->rh[lp->mat[j].row_nr] -= theta * lp->mat[j].value;
                    mpq_mul(*temp, *theta, *lp->mat[j].value);
                    mpq_sub(*lp->rh[lp->mat[j].row_nr], *lp->rh[lp->mat[j].row_nr], *temp);
                }
            }
        }
        invert(lp);
        lp->eta_valid = TRUE;
        failure = solvelp(lp); /* and solve again */
    }

    if(failure != OPTIMAL)
        debug_print(lp, "this problem has no solution, it is %s",
                    (failure == UNBOUNDED) ? "unbounded" : "infeasible");

    if(failure == INFEASIBLE && lp->verbose)
        fprintf(stderr, "level%4d INF\n", Level);

    if(failure == OPTIMAL) /* there is a good solution */
    {
        construct_solution(lp);

        /* because of reports of solution > upbo */
        /* check_solution(lp, upbo, lowbo); get too many hits ?? */

        debug_print(lp, "a solution was found");
        debug_print_solution(lp);

        /* if this solution is worse than the best sofar, this branch must die */

        /* if we can only have integer OF values, we might consider requiring to
           be at least 1 better than the best sofar, MB */

        if(lp->maximise)
            is_worse = (mpq_cmp(*lp->solution[0], *lp->best_solution[0]) <= 0);//lp->solution[0] <= lp->best_solution[0];
        else /* minimising! */
            is_worse = (mpq_cmp(*lp->solution[0], *lp->best_solution[0]) >= 0);//lp->solution[0] >= lp->best_solution[0];

        if(is_worse)
        {
            if(lp->verbose) {
                /*fprintf(stderr, "level%4d OPT NOB value %g bound %g\n",
                        Level, (double) lp->solution[0],
                        (double) lp->best_solution[0]);*/
                fprintf(stderr, "level%4d OPT NOB value ", Level);
                mpq_out_str(stderr, 10, *lp->solution[0]);
                fprintf(stderr, " bound ");
                mpq_out_str(stderr, 10, *lp->best_solution[0]);
                fprintf(stderr, "\n");
            }
            debug_print(lp, "but it was worse than the best sofar, discarded");
            Level--;

            mpq_clear(*temp);
            mpq_clear(*theta);
            mpq_clear(*tmpreal);
            return(MILP_FAIL);
        }

        /* check if solution contains enough ints */
        if(lp->bb_rule == FIRST_NI)
        {
            for(notint = 0, i = lp->rows + 1;
                i <= lp->sum && notint == 0;
                i++)
            {
                if(lp->must_be_int[i] && !is_int(lp, lp->solution[i])) {
                    if(mpq_equal(*lowbo[i], *upbo[i]))//lowbo[i] == upbo[i]) /* this var is already fixed */
                    {
                        //fprintf(stderr,
                                //"Warning: integer var %d is already fixed at %d, but has non-integer value %g\n",
                                //i - lp->rows, (int)lowbo[i],
                                //(double)lp->solution[i]);
                        //fprintf(stderr, "Perhaps the -e option should be used\n");
                        fprintf(stderr, "Warning: integer var %d is already fixed at %d, but has non-integer value ", i - lp->rows, (int) mpq_get_d(*lowbo[i]));
                        mpq_out_str(stderr, 10, *lp->solution[i]);
                        fprintf(stderr, "\nPerhaps the -e option should be used\n");

                    }
                    else
                        notint = i;
                }
            }
        }
        if(lp->bb_rule == RAND_NI)
        {
            int nr_not_int, select_not_int;
            nr_not_int = 0;

            for(i = lp->rows + 1; i <= lp->sum; i++)
                if(lp->must_be_int[i] && !is_int(lp, lp->solution[i]))
                    nr_not_int++;

            if(nr_not_int == 0)
                notint = 0;
            else
            {
                select_not_int = (rand() % nr_not_int) + 1;
                i = lp->rows + 1;
                while(select_not_int > 0)
                {
                    if(lp->must_be_int[i] && !is_int(lp, lp->solution[i]))
                        select_not_int--;
                    i++;
                }
                notint = i - 1;
            }
        }

        if(lp->verbose) {
            if(notint) {
                //fprintf(stderr, "level %3d OPT     value %f\n", Level,
                        //(double) lp->solution[0]);
                fprintf(stderr, "level %3d OPT     value \n", Level);
                mpq_out_str(stderr, 10, *lp->solution[0]);
                fprintf(stderr, "\n");
            }
            else {
                //fprintf(stderr, "level %3d OPT INT value %f\n", Level,
                        //(double) lp->solution[0]);
                fprintf(stderr, "level %3d OPT INT value ", Level);
                mpq_out_str(stderr, 10, *lp->solution[0]);
                fprintf(stderr, "\n");
            }
        }

        if(notint) /* there is at least one value not yet int */
        {
            /* set up two new problems */
            REAL   *new_upbo;
            REAL   *new_lowbo;
            REAL   new_bound;
            REAL   temp;
            short  *new_lower,*new_basis;
            int    *new_bas;
            int     resone, restwo;

            mpq_init(*new_bound);
            mpq_init(*temp);

            /* allocate room for them */
            MALLOC(new_upbo,  lp->sum + 1);
            MALLOC(new_lowbo, lp->sum + 1);
            MALLOC(new_lower, lp->sum + 1);
            MALLOC(new_basis, lp->sum + 1);
            MALLOC(new_bas,   lp->rows + 1);
            memcpy(new_upbo,  upbo,      (lp->sum + 1)  * sizeof(mpq_t));
            memcpy(new_lowbo, lowbo,     (lp->sum + 1)  * sizeof(mpq_t));
            memcpy(new_lower, lp->lower, (lp->sum + 1)  * sizeof(short));
            memcpy(new_basis, lp->basis, (lp->sum + 1)  * sizeof(short));
            memcpy(new_bas,   lp->bas,   (lp->rows + 1) * sizeof(int));



            if(lp->names_used) {
                //debug_print(lp, "not enough ints. Selecting var %s, val: %10.3g",
                            //lp->col_name[notint - lp->rows],
                            //(double) lp->solution[notint]);
               if(lp->debug){
                   fprintf(stderr, "not enough ints. Selecting var %s, val: ", lp->col_name[notint - lp->rows]);
                   mpq_out_str(stderr, 10, *lp->solution[notint]);
                   fprintf(stderr, "\n");
               }
            }
            else {
                //debug_print(lp,
                            //"not enough ints. Selecting Var [%5d], val: %10.3g",
                            //notint, (double) lp->solution[notint]);
                if(lp->debug){
                    fprintf(stderr, "not enough ints. Selecting Var [%5d], val: ", notint);
                    mpq_out_str(stderr, 10, *lp->solution[notint]);
                    fprintf(stderr, "\n");
                }
            }
            debug_print(lp, "current bounds:\n");
            debug_print_bounds(lp, upbo, lowbo);

            if(lp->floor_first)
            {
                //new_bound = ceil(lp->solution[notint]) - 1;
                mpq_set_d(*new_bound, ceil(mpq_get_d(*lp->solution[notint])) - 1);

                /* this bound might conflict */
                if(mpq_cmp(*new_bound, *lowbo[notint]) < 0)//new_bound < lowbo[notint])
                {
                    //debug_print(lp,
                                //"New upper bound value %g conflicts with old lower bound %g\n",
                                //(double)new_bound, (double)lowbo[notint]);
                    if(lp->debug){
                        fprintf(stderr, "New upper bound value ");
                        mpq_out_str(stderr, 10, *new_bound);
                        fprintf(stderr, " confilicts with old lower bound ");
                        mpq_out_str(stderr, 10, *lowbo[notint]);
                        fprintf(stderr, "\n");
                    }

                    resone = MILP_FAIL;
                }
                else /* bound feasible */
                {
                    check_if_less(new_bound, upbo[notint], lp->solution[notint]);
                    mpq_set(*new_upbo[notint], *new_bound);//new_upbo[notint] = new_bound;
                    debug_print(lp, "starting first subproblem with bounds:");
                    debug_print_bounds(lp, new_upbo, lowbo);
                    lp->eta_valid = FALSE;
                    resone = milpsolve(lp, new_upbo, lowbo, new_basis, new_lower,
                                       new_bas, TRUE);
                    lp->eta_valid = FALSE;
                }
                new_bound += 1;
                if(mpq_cmp(*new_bound, *upbo[notint]) > 0)//new_bound > upbo[notint])
                {
                    //debug_print(lp,
                                //"New lower bound value %g conflicts with old upper bound %g\n",
                                //(double)new_bound, (double)upbo[notint]);
                    if(lp->debug){
                        fprintf(stderr, "New lower bound value ");
                        mpq_out_str(stderr, 10, *new_bound);
                        fprintf(stderr, " conflicts with old upper bound ");
                        mpq_out_str(stderr, 10, *upbo[notint]);
                        fprintf(stderr, "\n");
                    }

                    restwo = MILP_FAIL;
                }
                else /* bound feasible */
                {
                    check_if_less(lowbo[notint], new_bound,
                                  lp->solution[notint]);
                    mpq_set(*new_lowbo[notint], *new_bound);//new_lowbo[notint] = new_bound;
                    debug_print(lp, "starting second subproblem with bounds:");
                    debug_print_bounds(lp, upbo, new_lowbo);
                    lp->eta_valid = FALSE;
                    restwo = milpsolve(lp, upbo, new_lowbo, new_basis, new_lower,
                                       new_bas, TRUE);
                    lp->eta_valid = FALSE;
                }
            }
            else /* take ceiling first */
            {
                /*VS - not sure if we need to be taking the ceiling here, but lp solver may need integers */
                mpq_set_d(*new_bound, ceil(mpq_get_d(*lp->solution[notint])));//new_bound = ceil(lp->solution[notint]);
                /* this bound might conflict */
                if(mpq_cmp(*new_bound, *upbo[notint]) > 0) //new_bound > upbo[notint])
                {
                    //debug_print(lp,
                                //"New lower bound value %g conflicts with old upper bound %g\n",
                                //(double)new_bound, (double)upbo[notint]);
                    if(lp->debug){
                        fprintf(stderr, "New lower bound value ");
                        mpq_out_str(stderr, 10, *new_bound);
                        fprintf(stderr, " conflicts with old upper bound ");
                        mpq_out_str(stderr, 10, *upbo[notint]);
                        fprintf(stderr, "\n");
                    }


                    resone = MILP_FAIL;
                }
                else /* bound feasible */
                {
                    check_if_less(lowbo[notint], new_bound,
                                  lp->solution[notint]);
                    new_lowbo[notint] = new_bound;
                    debug_print(lp, "starting first subproblem with bounds:");
                    debug_print_bounds(lp, upbo, new_lowbo);
                    lp->eta_valid = FALSE;
                    resone = milpsolve(lp, upbo, new_lowbo, new_basis, new_lower,
                                       new_bas, TRUE);
                    lp->eta_valid = FALSE;
                }
                mpq_set_ui(*temp, 1, 1);
                mpq_sub(*new_bound, *new_bound, *temp);//new_bound -= 1;
                if(mpq_cmp(*new_bound, *lowbo[notint]) < 0)//new_bound < lowbo[notint])
                {
                    //debug_print(lp,
                                //"New upper bound value %g conflicts with old lower bound %g\n",
                                //(double)new_bound, (double)lowbo[notint]);
                    if(lp->debug){
                        fprintf(stderr, "New upper bound value ");
                        mpq_out_str(stderr, 10, *new_bound);
                        fprintf(stderr, " conflicts with old lower bound ");
                        mpq_out_str(stderr, 10, *lowbo[notint]);
                        fprintf(stderr, "\n");
                    }
                    restwo = MILP_FAIL;
                }
                else /* bound feasible */
                {
                    check_if_less(new_bound, upbo[notint], lp->solution[notint]);
                    mpq_set(*new_upbo[notint], *new_bound);//new_upbo[notint] = new_bound;
                    debug_print(lp, "starting second subproblem with bounds:");
                    debug_print_bounds(lp, new_upbo, lowbo);
                    lp->eta_valid = FALSE;
                    restwo = milpsolve(lp, new_upbo, lowbo, new_basis, new_lower,
                                       new_bas, TRUE);
                    lp->eta_valid = FALSE;
                }
            }
            if(resone && restwo)	/* both failed and must have been infeasible */
                failure = INFEASIBLE;
            else
                failure = OPTIMAL;


            for(i = 0; i <= lp->sum; i++){
                mpq_clear(*new_upbo[i]);
                mpq_clear(*new_lowbo[i]);
            }

            free(new_upbo);
            free(new_lowbo);
            free(new_basis);
            free(new_lower);
            free(new_bas);
        }
        else /* all required values are int */
        {
            debug_print(lp, "--> valid solution found");

            if(lp->maximise)
                is_worse = (mpq_cmp(*lp->solution[0], *lp->best_solution[0]) < 0);//lp->solution[0] < lp->best_solution[0];
            else
                is_worse = (mpq_cmp(*lp->solution[0], *lp->best_solution[0]) > 0);//lp->solution[0] > lp->best_solution[0];

            if(!is_worse) /* Current solution better */
            {
                if(lp->debug || (lp->verbose && !lp->print_sol)) {
                    //fprintf(stderr,
                            //"*** new best solution: old: %g, new: %g ***\n",
                            //(double) lp->best_solution[0], (double) lp->solution[0]);
                    fprintf(stderr, "*** new best solution: old: ");
                    mpq_out_str(stderr, 10, *lp->best_solution[0]);
                    fprintf(stderr, ", new: ");
                    mpq_out_str(stderr, 10, *lp->solution[0]);
                    fprintf(stderr, " ***\n");
                }
                memcpy(lp->best_solution, lp->solution,
                       (lp->sum + 1) * sizeof(mpq_t));
                calculate_duals(lp);

                if(lp->print_sol)
                    print_solution(lp);

                if(lp->break_at_int)
                {
                    if(lp->maximise && mpq_cmp(*lp->best_solution[0], *lp->break_value) > 0)//(lp->best_solution[0] > lp->break_value))
                        Break_bb = TRUE;

                    if(!lp->maximise && mpq_cmp(*lp->best_solution[0], *lp->break_value) < 0)//(lp->best_solution[0] < lp->break_value))
                        Break_bb = TRUE;
                }
            }
        }
    }

    Level--;

    /* failure can have the values OPTIMAL, UNBOUNDED and INFEASIBLE. */

    mpq_clear(*theta);
    mpq_clear(*tmpreal);
    mpq_clear(*temp);
    mpz_clear(tmpInt);

    return(failure);
} /* milpsolve */


int solve(lprec *lp)
{
    int result, i;

    REAL temp;
    mpq_init(*temp);
    mpq_neg(*temp, *lp->infinite); //set up for comparison below

    lp->total_iter  = 0;
    lp->max_level   = 1;
    lp->total_nodes = 0;

    if(isvalid(lp))
    {
        if(lp->maximise && mpq_equal(*lp->obj_bound, *lp->infinite))//lp->obj_bound == lp->infinite)
            mpq_neg(*lp->best_solution[0], *lp->infinite);//lp->best_solution[0] = -lp->infinite;
        else if(!lp->maximise && mpq_equal(*lp->obj_bound, *temp))//lp->obj_bound == -lp->infinite)
            mpq_set(*lp->best_solution[0], *lp->infinite);//lp->best_solution[0] = lp->infinite;
        else
            mpq_set(*lp->best_solution[0], *lp->obj_bound);//lp->best_solution[0] = lp->obj_bound;

        Level = 0;

        if(!lp->basis_valid)
        {
            for(i = 0; i <= lp->rows; i++)
            {
                lp->basis[i] = TRUE;
                lp->bas[i]   = i;
            }

            for(i = lp->rows + 1; i <= lp->sum; i++)
                lp->basis[i] = FALSE;

            for(i = 0; i <= lp->sum; i++)
                lp->lower[i] = TRUE;

            lp->basis_valid = TRUE;
        }

        lp->eta_valid = FALSE;
        Break_bb      = FALSE;
        result        = milpsolve(lp, lp->orig_upbo, lp->orig_lowbo, lp->basis,
                                  lp->lower, lp->bas, FALSE);
        mpq_clear(*temp);
        return(result);
    }

    /* if we get here, isvalid(lp) failed. I suggest we return FAILURE */
    fprintf(stderr, "Error, the current LP seems to be invalid\n");
    mpq_clear(*temp);
    return(FAILURE);
} /* solve */

int lag_solve(lprec *lp, REAL start_bound, int num_iter, short verbose)
{
    int i, j, result, citer;
    short status, OrigFeas, AnyFeas, same_basis;
    REAL *OrigObj;
    REAL *ModObj;
    REAL *SubGrad;
    REAL *BestFeasSol;
    REAL Zub;
    REAL Zlb;
    REAL Ztmp;
    REAL pie;
    REAL rhsmod;
    REAL Step;
    REAL SqrsumSubGrad;
    REAL temp;
    REAL point95; /*Used as a constant to multiply pie by. */
    REAL onepoint05; /*Used as a constant to multiply pie by. */
    int   *old_bas;
    short *old_lower;

    mpq_init(*Zub);
    mpq_init(*Zlb);
    mpq_init(*Ztmp);
    mpq_init(*pie);
    mpq_init(*rhsmod);
    mpq_init(*Step);
    mpq_init(*SqrsumSubGrad);
    mpq_init(*temp);
    mpq_init(*point95);
    mpq_init(*onepoint05);

    /* allocate mem */
    MALLOC(OrigObj, lp->columns + 1);
    CALLOC(ModObj, lp->columns + 1);
    CALLOC(SubGrad, lp->nr_lagrange);
    CALLOC(BestFeasSol, lp->sum + 1);
    MALLOCCPY(old_bas, lp->bas, lp->rows + 1);
    MALLOCCPY(old_lower, lp->lower, lp->sum + 1);

    /*Allocate REALs */
    for(i = 0; i <= lp->columns; i++){
        mpq_init(*OrigObj[i]);
        mpq_init(*ModObj[i]);
    }
    for(i = 0; i < lp->nr_lagrange; i++)
        mpq_init(*SubGrad[i]);

    for(i = 0; i <= lp->sum; i++)
        mpq_init(*BestFeasSol[i]);

    get_row(lp, 0, OrigObj);

    mpq_set_d(*pie, 2.0);//pie = 2;
    mpq_set_d(*point95, 0.95);
    mpq_set_d(*onepoint05, 1.95);

    if(lp->maximise)
    {
        mpq_set_d(*Zub, DEF_INFINITE);//Zub = DEF_INFINITE;
        mpq_set(*Zlb, *start_bound);//Zlb = start_bound;
    }
    else
    {
        mpq_set_d(*Zlb, -DEF_INFINITE);//Zlb = -DEF_INFINITE;
        mpq_set(*Zub, *start_bound);//Zub = start_bound;
    }
    status   = RUNNING;
    mpq_set_ui(*Step, 1, 1); //Step     = 1;
    OrigFeas = FALSE;
    AnyFeas  = FALSE;
    citer    = 0;

    for(i = 0 ; i < lp->nr_lagrange; i++)
        mpq_set_ui(*lp->lambda[i], 0, 1);//lp->lambda[i] = 0;

    while(status == RUNNING)
    {
        citer++;

        for(i = 1; i <= lp->columns; i++)
        {
            mpq_set(*ModObj[i], *OrigObj[i]);//ModObj[i] = OrigObj[i];
            for(j = 0; j < lp->nr_lagrange; j++)
            {
                if(lp->maximise) {
                    //ModObj[i] -= lp->lambda[j] * lp->lag_row[j][i];
                    mpq_mul(*temp, *lp->lambda[j], *lp->lag_row[j][i]);
                    mpq_sub(*ModObj[i], *ModObj[i], *temp);
                }
                else {
                    //ModObj[i] += lp->lambda[j] * lp->lag_row[j][i];
                    mpq_mul(*temp, *lp->lambda[j], *lp->lag_row[j][i]);
                    mpq_add(*ModObj[i], *ModObj[i], *temp);
                }
            }
        }
        for(i = 1; i <= lp->columns; i++)
        {
            set_mat(lp, 0, i, ModObj[i]);
        }
        mpq_set_ui(*rhsmod, 0, 1); //rhsmod = 0;
        for(i = 0; i < lp->nr_lagrange; i++)
            if(lp->maximise) {
                //rhsmod += lp->lambda[i] * lp->lag_rhs[i];
                mpq_mul(*temp, *lp->lambda[i], *lp->lag_rhs[i]);
                mpq_add(*rhsmod, *rhsmod, *temp);
            }
            else {
                //rhsmod -= lp->lambda[i] * lp->lag_rhs[i];
                mpq_mul(*temp, *lp->lambda[i], *lp->lag_rhs[i]);
                mpq_sub(*rhsmod, *rhsmod, *temp);
            }


        if(verbose)
        {
            //fprintf(stderr, "Zub: %10f Zlb: %10f Step: %10f pie: %10f Feas %d\n",
                    //(double)Zub, (double)Zlb, (double)Step, (double)pie,
                    //OrigFeas);
            fprintf(stderr, "Zub: ");
            mpq_out_str(stderr, 10, *Zub);
            fprintf(stderr, " Zlb: ");
            mpq_out_str(stderr, 10, *Zlb);
            fprintf(stderr, " Step: ");
            mpq_out_str(stderr, 10, *Step);
            fprintf(stderr, " pie: ");
            mpq_out_str(stderr, 10, *pie);
            fprintf(stderr, " Feas %d\n", OrigFeas);

            for(i = 0; i < lp->nr_lagrange; i++) {
                //fprintf(stderr, "%3d SubGrad %10f lambda %10f\n", i,
                        //(double) SubGrad[i], (double) lp->lambda[i]);

                fprintf(stderr, "%3d SubGrad ", i);
                mpq_out_str(stderr, 10, *SubGrad[i]);
                fprintf(stderr, " lambda ");
                mpq_out_str(stderr, 10, *lp->lambda[i]);
                fprintf(stderr, "\n");

            }
        }

        if(verbose && lp->sum < 20)
            print_lp(lp);

        result = solve(lp);

        if(verbose && lp->sum < 20)
        {
            print_solution(lp);
        }

        same_basis = TRUE;
        i = 1;
        while(same_basis && i < lp->rows)
        {
            same_basis = (old_bas[i] == lp->bas[i]);
            i++;
        }
        i = 1;
        while(same_basis && i < lp->sum)
        {
            same_basis=(old_lower[i] == lp->lower[i]);
            i++;
        }
        if(!same_basis)
        {
            memcpy(old_lower, lp->lower, (lp->sum+1) * sizeof(short));
            memcpy(old_bas, lp->bas, (lp->rows+1) * sizeof(int));
            mpq_mul(*pie, *pie, *point95);//pie *= 0.95;
        }

        if(verbose)
            fprintf(stderr, "result: %d  same basis: %d\n", result, same_basis);

        if(result == UNBOUNDED)
        {
            for(i = 1; i <= lp->columns; i++)
                mpq_out_str(stderr, 10, *ModObj[i]);//fprintf(stderr, "%5f ", (double)ModObj[i]);
            exit(EXIT_FAILURE);
        }

        if(result == FAILURE)
            status = FAILURE;

        if(result == INFEASIBLE)
            status = INFEASIBLE;

        mpq_set_ui(*SqrsumSubGrad, 0, 1);//SqrsumSubGrad = 0;
        for(i = 0; i < lp->nr_lagrange; i++)
        {
            mpq_neg(*SubGrad[i], *lp->lag_rhs[i]);//SubGrad[i]= -lp->lag_rhs[i];
            for(j = 1; j <= lp->columns; j++) {
                //SubGrad[i] += lp->best_solution[lp->rows + j] * lp->lag_row[i][j];
                mpq_mul(*temp, *lp->best_solution[lp->rows + j], *lp->lag_row[i][j]);
                mpq_add(*SubGrad[i], *SubGrad[i], *temp);
            }
            //SqrsumSubGrad += SubGrad[i] * SubGrad[i];
            mpq_mul(*temp, *SubGrad[i], *SubGrad[i]);
            mpq_add(*SqrsumSubGrad, *SqrsumSubGrad, *temp);
        }

        OrigFeas = TRUE;
        for(i = 0; i < lp->nr_lagrange; i++)
            if(lp->lag_con_type[i])
            {
                mpq_abs(*temp, *SubGrad[i]);
                if(mpq_cmp(*temp, *lp->epsb) > 0)//my_abs(SubGrad[i]) > lp->epsb)
                    OrigFeas = FALSE;
            }
            else if(mpq_cmp(*SubGrad[i], *lp->epsb) > 0)//SubGrad[i] > lp->epsb)
                OrigFeas = FALSE;

        if(OrigFeas)
        {
            AnyFeas = TRUE;
            mpq_set_ui(*Ztmp, 0, 1); //Ztmp = 0;
            for(i = 1; i <= lp->columns; i++) {
                //Ztmp += lp->best_solution[lp->rows + i] * OrigObj[i];
                mpq_mul(*temp, *lp->best_solution[lp->rows + i], *OrigObj[i]);
                mpq_add(*Ztmp, *Ztmp, *temp);
            }
            if((lp->maximise) && (mpq_cmp(*Ztmp, *Zlb) > 0))//(Ztmp > Zlb))
            {
                mpq_set(*Zlb, *Ztmp);//Zlb = Ztmp;
                for(i = 1; i <= lp->sum; i++)
                    mpq_set(*BestFeasSol[i], *lp->best_solution[i]);//BestFeasSol[i] = lp->best_solution[i];
                mpq_set(*BestFeasSol[0], *Zlb);//BestFeasSol[0] = Zlb;
                if(verbose) {
                   //fprintf(stderr, "Best feasible solution: %f\n", (double) Zlb);
                    fprintf(stderr, "Best feasible solution: ");
                    mpq_out_str(stderr, 10, *Zlb);
                    fprintf(stderr, "\n");
                }
            }
            else if(mpq_cmp(*Ztmp, *Zub) < 0) //Ztmp < Zub)
            {
                mpq_set(*Zub, *Ztmp); //Zub = Ztmp;
                for(i = 1; i <= lp->sum; i++)
                    mpq_set(*BestFeasSol[i], *lp->best_solution[i]);//BestFeasSol[i] = lp->best_solution[i];
                mpq_set(*BestFeasSol[0], *Zub);//BestFeasSol[0] = Zub;
                if(verbose) {
                    //fprintf(stderr, "Best feasible solution: %f\n", (double) Zub);
                    fprintf(stderr, "Best feasible solution: ");
                    mpq_out_str(stderr, 10, *Zub);
                    fprintf(stderr, "\n");
                }
            }
        }

        if(lp->maximise) {
            mpq_add(*temp, *rhsmod, *lp->best_solution[0]);
            mpq_set(*Zub, *(my_mpq_min(Zub, temp)));//Zub = my_min(Zub, rhsmod + lp->best_solution[0]);
        }
        else {
            //Zlb = my_max(Zlb, rhsmod + lp->best_solution[0]);
            mpq_add(*temp, *rhsmod, *lp->best_solution[0]);
            mpq_set(*Zlb, *(my_mpq_max(Zlb, temp)));
        }

        mpq_sub(*temp, *Zub, *Zlb);
        mpq_abs(*temp, *temp);
        //if(my_abs(Zub-Zlb)<0.001)
        if(mpq_get_d(*temp) < 0.001) /*VS - ugly constant. Why this number? */
        {
            status = OPTIMAL;
        }
        //Step = pie * ((1.05*Zub) - Zlb) / SqrsumSubGrad;
        mpq_mul(*temp, *onepoint05, *Zub); // (1.05 * Zub)
        mpq_sub(*temp, *temp, *Zlb); //(1.05 * Zub) - Zlb
        mpq_mul(*temp, *pie, *temp); // pie * ((1.05 * Zub) - Zlb)
        mpq_div(*Step, *temp, *SqrsumSubGrad);

        for(i = 0; i < lp->nr_lagrange; i++)
        {
            //lp->lambda[i] += Step * SubGrad[i];
            mpq_mul(*temp, *Step, *SubGrad[i]);
            mpq_add(*lp->lambda[i], *lp->lambda[i], *temp);

            if(!lp->lag_con_type[i] && mpq_sgn(*lp->lambda[i]) < 0)//lp->lambda[i] < 0)
                mpq_set_ui(*lp->lambda[i], 0, 1);//lp->lambda[i] = 0;
        }

        if(citer == num_iter && status==RUNNING) {
            if(AnyFeas)
                status = FEAS_FOUND;
            else
                status = NO_FEAS_FOUND;
        }
    }

    for(i = 0; i <= lp->sum; i++)
        mpq_set(*lp->best_solution[i], *BestFeasSol[i]);//lp->best_solution[i] = BestFeasSol[i];

    for(i = 1; i <= lp->columns; i++)
        set_mat(lp, 0, i, OrigObj[i]);

    if(lp->maximise)
        mpq_set(*lp->lag_bound, *Zub);//lp->lag_bound = Zub;
    else
        mpq_set(*lp->lag_bound, *Zlb);//lp->lag_bound = Zlb;


    mpq_clear(*Zub);
    mpq_clear(*Zlb);
    mpq_clear(*Ztmp);
    mpq_clear(*pie);
    mpq_clear(*rhsmod);
    mpq_clear(*Step);
    mpq_clear(*SqrsumSubGrad);
    mpq_clear(*temp);
    mpq_clear(*point95);
    mpq_clear(*onepoint05);


    for(i = 0; i <= lp->columns; i++){
        mpq_clear(*OrigObj[i]);
        mpq_clear(*ModObj[i]);
    }
    for(i = 0; i < lp->nr_lagrange; i++)
        mpq_clear(*SubGrad[i]);

    for(i = 0; i <= lp->sum; i++)
        mpq_clear(*BestFeasSol[i]);

    free(BestFeasSol);
    free(SubGrad);
    free(OrigObj);
    free(ModObj);
    free(old_bas);
    free(old_lower);

    return(status);
}

