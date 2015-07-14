#include "lpkit.h"
#include "lpglob.h"
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <gmp.h>
#define HASHSIZE 10007

/* Globals */
int     Rows;
int     Columns;
int     Sum;
int     Non_zeros;
int     Level;

REAL	Trej;

short   Maximise;
REAL    Extrad;

int     Warn_count; /* used in CHECK version of rounding macro */

void error(char *format, ...)
{
    va_list ap;

    va_start(ap, format);
    vfprintf(stderr, format, ap);
    fputc('\n', stderr);
    va_end(ap);

    exit(EXIT_FAILURE);
}

lprec *make_lp(int rows, int columns)
{
    lprec *newlp;
    int i, sum;

    if(rows < 0 || columns < 0)
        error("rows < 0 or columns < 0");

    sum = rows + columns;

    CALLOC(newlp, 1);

    strcpy(newlp->lp_name, "unnamed");

    newlp->verbose = FALSE;
    newlp->print_duals = FALSE;
    newlp->print_sol = FALSE;
    newlp->debug = FALSE;
    newlp->print_at_invert = FALSE;
    newlp->trace = FALSE;

    newlp->rows = rows;
    newlp->columns = columns;
    newlp->sum = sum;
    newlp->rows_alloc = rows;
    newlp->columns_alloc = columns;
    newlp->sum_alloc = sum;
    newlp->names_used = FALSE;

    mpq_init(newlp->obj_bound);
    mpq_set_d(newlp->obj_bound, DEF_INFINITE);

    mpq_init(newlp->infinite);
    mpq_set_d(newlp->infinite,DEF_INFINITE);

    mpq_init(newlp->epsilon);
    mpq_set_d(newlp->epsilon,DEF_EPSILON);

    mpq_init(newlp->epsb);
    mpq_set_d(newlp->epsb, DEF_EPSB);

    mpq_init(newlp->epsd);
    mpq_set_d(newlp->epsd, DEF_EPSD);

    mpq_init(newlp->epsel);
    mpq_set_d(newlp->epsel, DEF_EPSEL);
    newlp->non_zeros = 0;
    newlp->mat_alloc = 1;
    CALLOC(newlp->mat, newlp->mat_alloc);
    CALLOC(newlp->col_no, newlp->mat_alloc + 1);
    CALLOC(newlp->col_end, columns + 1);
    CALLOC(newlp->row_end, rows + 1);
    newlp->row_end_valid = FALSE;
    CALLOC(newlp->orig_rh, rows + 1);
    CALLOC(newlp->rh, rows + 1);
    CALLOC(newlp->rhs, rows + 1);
    CALLOC(newlp->must_be_int, sum + 1);
    for(i = 0; i <= sum; i++)
        newlp->must_be_int[i]=FALSE;
    CALLOC(newlp->orig_upbo, sum + 1);
    for(i = 0; i <= sum; i++)
        mpq_set(newlp->orig_upbo[i],newlp->infinite);//newlp->orig_upbo[i]=newlp->infinite;
    CALLOC(newlp->upbo, sum + 1);
    CALLOC(newlp->orig_lowbo, sum + 1);
    CALLOC(newlp->lowbo, sum + 1);

    newlp->basis_valid = TRUE;
    CALLOC(newlp->bas, rows+1);
    CALLOC(newlp->basis, sum + 1);
    CALLOC(newlp->lower, sum + 1);

    for(i = 0; i <= rows; i++)
    {
        newlp->bas[i] = i;
        newlp->basis[i] = TRUE;
    }

    for(i = rows + 1; i <= sum; i++)
        newlp->basis[i] = FALSE;


    for(i = 0 ; i <= sum; i++)
        newlp->lower[i] = TRUE;

    newlp->eta_valid = TRUE;
    newlp->eta_size = 0;
    newlp->eta_alloc = INITIAL_MAT_SIZE;
    newlp->max_num_inv = DEFNUMINV;

    newlp->nr_lagrange = 0;

    CALLOC(newlp->eta_value, newlp->eta_alloc);
    CALLOC(newlp->eta_row_nr, newlp->eta_alloc);
    /* +1 reported by Christian Rank */
    CALLOC(newlp->eta_col_end, newlp->rows_alloc + newlp->max_num_inv + 1);

    newlp->bb_rule = FIRST_NI;
    newlp->break_at_int = FALSE;
    mpq_set_ui(newlp->break_value, 0, 1);//newlp->break_value = 0;

    newlp->iter = 0;
    newlp->total_iter = 0;

    CALLOC(newlp->solution, sum + 1);
    CALLOC(newlp->best_solution, sum + 1);
    CALLOC(newlp->duals, rows + 1);

    newlp->maximise = FALSE;
    newlp->floor_first = TRUE;

    newlp->scaling_used = FALSE;
    newlp->columns_scaled = FALSE;

    CALLOC(newlp->ch_sign, rows + 1);

    for(i = 0; i <= rows; i++)
        newlp->ch_sign[i] = FALSE;

    newlp->valid = FALSE;

    /* create two hash tables for names */
    newlp->rowname_hashtab = create_hash_table(HASHSIZE);
    newlp->colname_hashtab = create_hash_table(HASHSIZE);

    return(newlp);
}

void delete_lp(lprec *lp)
{
    int i;

    if(lp->names_used)
    {
        free(lp->row_name);
        free(lp->col_name);
    }
    mpq_clear(lp->obj_bound);

    mpq_clear(lp->infinite);

    mpq_clear(lp->epsilon);

    mpq_clear(lp->epsb);

    mpq_clear(lp->epsd);

    mpq_clear(lp->epsel);

    free(lp->mat);
    free(lp->col_no);
    free(lp->col_end);
    free(lp->row_end);
    free(lp->orig_rh);
    free(lp->rh);
    free(lp->rhs);
    free(lp->must_be_int);
    free(lp->orig_upbo);
    free(lp->orig_lowbo);
    free(lp->upbo);
    free(lp->lowbo);
    free(lp->bas);
    free(lp->basis);
    free(lp->lower);
    free(lp->eta_value);
    free(lp->eta_row_nr);
    free(lp->eta_col_end);
    free(lp->solution);
    free(lp->best_solution);
    free(lp->duals);
    free(lp->ch_sign);
    if(lp->scaling_used)
        free(lp->scale);
    if(lp->nr_lagrange > 0)
    {
        free(lp->lag_rhs);
        free(lp->lambda);
        free(lp->lag_con_type);
        for(i = 0; i < lp->nr_lagrange; i++)
            free(lp->lag_row[i]);
        free(lp->lag_row);
    }

    free_hash_table(lp->rowname_hashtab);
    free_hash_table(lp->colname_hashtab);

    free(lp);

}

lprec *copy_lp(lprec *lp)
{
    lprec *newlp;
    int i, rowsplus, colsplus, sumplus;

    rowsplus=lp->rows_alloc+1;
    colsplus=lp->columns_alloc+1;
    sumplus=lp->sum_alloc+1;

    MALLOCCPY(newlp, lp, 1); /* copy all non pointers */

    if(newlp->names_used)
    {
        MALLOCCPY(newlp->col_name, lp->col_name, colsplus);
        MALLOCCPY(newlp->row_name, lp->row_name, rowsplus);
    }

    MALLOCCPY(newlp->mat, lp->mat, newlp->mat_alloc);
    MALLOCCPY(newlp->col_end, lp->col_end, colsplus);
    MALLOCCPY(newlp->col_no, lp->col_no, newlp->mat_alloc + 1);
    MALLOCCPY(newlp->row_end, lp->row_end, rowsplus);
    MALLOCCPY(newlp->orig_rh, lp->orig_rh, rowsplus);
    MALLOCCPY(newlp->rh, lp->rh, rowsplus);
    MALLOCCPY(newlp->rhs, lp->rhs, rowsplus);
    MALLOCCPY(newlp->must_be_int, lp->must_be_int, sumplus);
    MALLOCCPY(newlp->orig_upbo, lp->orig_upbo, sumplus);
    MALLOCCPY(newlp->orig_lowbo, lp->orig_lowbo, sumplus);
    MALLOCCPY(newlp->upbo, lp->upbo, sumplus);
    MALLOCCPY(newlp->lowbo, lp->lowbo, sumplus);
    MALLOCCPY(newlp->bas, lp->bas, rowsplus);
    MALLOCCPY(newlp->basis, lp->basis, sumplus);
    MALLOCCPY(newlp->lower, lp->lower, sumplus);
    MALLOCCPY(newlp->eta_value, lp->eta_value, lp->eta_alloc);
    MALLOCCPY(newlp->eta_row_nr, lp->eta_row_nr, lp->eta_alloc);
    MALLOCCPY(newlp->eta_col_end, lp->eta_col_end,
              lp->rows_alloc + lp->max_num_inv + 1);
    MALLOCCPY(newlp->solution, lp->solution, sumplus);
    MALLOCCPY(newlp->best_solution, lp->best_solution, sumplus);
    MALLOCCPY(newlp->duals, lp->duals, rowsplus);
    MALLOCCPY(newlp->ch_sign, lp->ch_sign, rowsplus);

    if(newlp->scaling_used)
    MALLOCCPY(newlp->scale, lp->scale, sumplus);

    if(newlp->nr_lagrange > 0)
    {
        MALLOCCPY(newlp->lag_rhs, lp->lag_rhs, newlp->nr_lagrange);
        MALLOCCPY(newlp->lambda, lp->lambda, newlp->nr_lagrange);
        MALLOCCPY(newlp->lag_con_type, lp->lag_con_type, newlp->nr_lagrange);
        MALLOC(newlp->lag_row, newlp->nr_lagrange);
        for(i = 0; i < newlp->nr_lagrange; i++)
        MALLOCCPY(newlp->lag_row[i], lp->lag_row[i], colsplus);
    }
    return(newlp);
}

void inc_mat_space(lprec *lp, int maxextra)
{
    if(lp->non_zeros + maxextra >= lp->mat_alloc)
    {
        /* let's allocate at least INITIAL_MAT_SIZE  entries */
        if(lp->mat_alloc < INITIAL_MAT_SIZE)
            lp->mat_alloc = INITIAL_MAT_SIZE;

        /* increase the size by 50% each time it becomes too small */
        while(lp->non_zeros + maxextra >= lp->mat_alloc)
            lp->mat_alloc *= 1.5;

        REALLOC(lp->mat, lp->mat_alloc);
        REALLOC(lp->col_no, lp->mat_alloc + 1);
    }
}

void inc_row_space(lprec *lp)
{
    if(lp->rows > lp->rows_alloc)
    {
        lp->rows_alloc=lp->rows+10;
        lp->sum_alloc=lp->rows_alloc+lp->columns_alloc;
        REALLOC(lp->orig_rh, lp->rows_alloc + 1);
        REALLOC(lp->rh, lp->rows_alloc + 1);
        REALLOC(lp->rhs, lp->rows_alloc + 1);
        REALLOC(lp->orig_upbo, lp->sum_alloc + 1);
        REALLOC(lp->upbo, lp->sum_alloc + 1);
        REALLOC(lp->orig_lowbo, lp->sum_alloc + 1);
        REALLOC(lp->lowbo, lp->sum_alloc + 1);
        REALLOC(lp->solution, lp->sum_alloc + 1);
        REALLOC(lp->best_solution, lp->sum_alloc + 1);
        REALLOC(lp->row_end, lp->rows_alloc + 1);
        REALLOC(lp->basis, lp->sum_alloc + 1);
        REALLOC(lp->lower, lp->sum_alloc + 1);
        REALLOC(lp->must_be_int, lp->sum_alloc + 1);
        REALLOC(lp->bas, lp->rows_alloc + 1);
        REALLOC(lp->duals, lp->rows_alloc + 1);
        REALLOC(lp->ch_sign, lp->rows_alloc + 1);
        REALLOC(lp->eta_col_end, lp->rows_alloc + lp->max_num_inv + 1);
        if(lp->names_used)
        REALLOC(lp->row_name, lp->rows_alloc + 1);
        if(lp->scaling_used)
        REALLOC(lp->scale, lp->sum_alloc + 1);
    }
}

void inc_col_space(lprec *lp)
{
    if(lp->columns >= lp->columns_alloc)
    {
        lp->columns_alloc=lp->columns+10;
        lp->sum_alloc=lp->rows_alloc+lp->columns_alloc;
        REALLOC(lp->must_be_int, lp->sum_alloc + 1);
        REALLOC(lp->orig_upbo, lp->sum_alloc + 1);
        REALLOC(lp->upbo, lp->sum_alloc + 1);
        REALLOC(lp->orig_lowbo, lp->sum_alloc + 1);
        REALLOC(lp->lowbo, lp->sum_alloc + 1);
        REALLOC(lp->solution, lp->sum_alloc + 1);
        REALLOC(lp->best_solution, lp->sum_alloc + 1);
        REALLOC(lp->basis, lp->sum_alloc + 1);
        REALLOC(lp->lower, lp->sum_alloc + 1);
        if(lp->names_used)
        REALLOC(lp->col_name, lp->columns_alloc + 1);
        if(lp->scaling_used)
        REALLOC(lp->scale, lp->sum_alloc + 1);
        REALLOC(lp->col_end, lp->columns_alloc + 1);
    }
}

void set_mat(lprec *lp, int Row, int Column, REAL Value)
{
    int elmnr, lastelm, i;

    REAL temp;
    mpq_init(temp);
    /* This function is very inefficient if used to add new matrix entries in
       other places than at the end of the matrix. OK for replacing existing
       non-zero values */


    if(Row > lp->rows || Row < 0)
        error("Row out of range");
    if(Column > lp->columns || Column < 1)
        error("Column out of range");

    /* scaling is performed twice? MB */
    if(lp->scaling_used)
        /*Value *= lp->scale[Row] * lp->scale[lp->rows + Column];*/
        /*VS Above translated to GMP rational arithmetic:*/
        mpq_mul(temp, lp->scale[Row], lp->scale[lp->rows + Column]);
        mpq_mul(Value, temp, Value);

    if (lp->basis[Column] == TRUE && Row > 0)
        lp->basis_valid = FALSE;
    lp->eta_valid = FALSE;

    /* find out if we already have such an entry */
    elmnr = lp->col_end[Column - 1];
    while((elmnr < lp->col_end[Column]) && (lp->mat[elmnr].row_nr != Row))
        elmnr++;

    mpq_abs(temp, Value);
    if((elmnr != lp->col_end[Column]) && (lp->mat[elmnr].row_nr == Row))
        /* there is an existing entry */
    {
        /*VS Replace call to my_abs(Value) to setting temp to abs(Value) and comparing with that.*/
        if (mpq_cmp(*temp, *lp->epsilon)) /* we replace it by something non-zero */
        {
            if (lp->scaling_used) {
                if (lp->ch_sign[Row]) {
                    /*lp->mat[elmnr].value = -Value * lp->scale[Row] * lp->scale[Column];*/
                    /*VS Above translated to GMP rational arithmetic*/
                    mpq_neg(temp, Value);
                    mpq_mul(temp, temp, lp->scale[Row]);
                    mpq_mul(temp, temp, lp->scale[Column]);
                    mpq_set(lp->mat[elmnr].value, temp);
                }
                else {
                    /*lp->mat[elmnr].value = Value * lp->scale[Row] * lp->scale[Column];*/
                    /*VS Above translated to GMP rational arithmetic */
                    mpq_mul(temp, Value, lp->scale[Row]);
                    mpq_mul(temp, temp, lp->scale[Column]);
                    mpq_set(lp->mat[elmnr].value, temp);
                }
            }
            else /* no scaling */
            {
                if (lp->ch_sign[Row]) {
                    /*lp->mat[elmnr].value = -Value;*/
                    /*VS Above translated to GMP rational arithmetic */
                    mpq_neg(temp, Value);
                    mpq_set(lp->mat[elmnr].value, temp);
                }

                else {
                    /*lp->mat[elmnr].value = Value;*/
                    /*VS Above translated to GMP rational arithmetic */
                    mpq_set(lp->mat[elmnr].value, Value);
                }
            }
        }
        else /* setting existing non-zero entry to zero. Remove the entry */
        {
            /* this might remove an entire column, or leave just a bound. No
                   nice solution for that yet */

            /* Shift the matrix */
            lastelm = lp->non_zeros;
            for (i = elmnr; i < lastelm; i++)
                lp->mat[i] = lp->mat[i + 1];
            for (i = Column; i <= lp->columns; i++)
                lp->col_end[i]--;

            lp->non_zeros--;
        }
    }
    else if(mpq_cmp(temp, lp->epsilon))
        /* no existing entry. make new one only if not nearly zero */
    {
        /* check if more space is needed for matrix */
        inc_mat_space(lp, 1);

        /* Shift the matrix */
        lastelm = lp->non_zeros;
        for(i = lastelm; i > elmnr ; i--)
            lp->mat[i] = lp->mat[i - 1];
        for(i = Column; i <= lp->columns; i++)
            lp->col_end[i]++;

        /* Set new element */
        lp->mat[elmnr].row_nr = Row;

        if (lp->scaling_used)
        {
            if(lp->ch_sign[Row]) {
                /*lp->mat[elmnr].value = -Value * lp->scale[Row] * lp->scale[Column];*/
                /*VS Above translated to GMP rational arithmetic */
                mpq_neg(temp, Value);
                mpq_mul(temp, temp, lp->scale[Row]);
                mpq_mul(temp, temp, lp->scale[Column]);
                mpq_set(lp->mat[elmnr].value, temp);
            }
            else {
                /*lp->mat[elmnr].value = Value * lp->scale[Row] * lp->scale[Column];*/
                /*VS Above translated to GMP rational arithmetic */
                mpq_mul(temp, Value, lp->scale[Row]);
                mpq_mul(temp, temp, lp->scale[Column]);
                mpq_set(lp->mat[elmnr].value, temp);
            }
        }
        else /* no scaling */
        {
            if(lp->ch_sign[Row]) {
                /*lp->mat[elmnr].value = -Value;*/
                /*VS above translated to GMP rational arithmetic */
                mpq_neg(temp, Value);
                mpq_set(lp->mat[elmnr].value, temp);
            }
            else {
                /*lp->mat[elmnr].value = Value;*/
                /*VS above translated to GMP rational arithmetic */
                mpq_set(lp->mat[elmnr].value, Value);
            }
        }

        lp->row_end_valid = FALSE;

        lp->non_zeros++;
    }
    mpq_clear(temp);
}

void set_obj_fn(lprec *lp, REAL *row)
{
    int i;
    for(i = 1; i <= lp->columns; i++)
        set_mat(lp, 0, i, row[i]);
}

void str_set_obj_fn(lprec *lp, char *row)
{
    int  i;
    double *arow;
    REAL arow_REAL[lp->columns+1]; /*VS array of rationals */
    char *p, *newp;
    CALLOC(arow, lp->columns + 1);

    p = row;
    for(i = 1; i <= lp->columns; i++)
    {
        arow[i] = (double) strtod(p, &newp);
        /*VS We convert the read double to rational */
        mpq_init(arow_REAL[i]);
        mpq_set_d(arow_REAL[i], arow[i]);
        if(p == newp)
            error("Bad string in str_set_obj_fn");
        else
            p = newp;
    }
    set_obj_fn(lp, arow_REAL);
    free(arow);
    for(i = 1; i <= lp->columns; i++)
        mpq_clear(arow_REAL[i]);
}


void add_constraint(lprec *lp, REAL *row, short constr_type, REAL rh)
{
    matrec *newmat;
    int  i, j;
    int  elmnr;
    int  stcol;
    int  *addtoo;
    REAL temp;
    mpq_init(temp);

    MALLOC(addtoo, lp->columns + 1);

    for(i = 1; i <= lp->columns; i++)
        if(row[i]!=0)
        {
            addtoo[i]=TRUE;
            lp->non_zeros++;
        }
        else
            addtoo[i]=FALSE;

    MALLOC(newmat, lp->non_zeros);
    inc_mat_space(lp, 0);
    lp->rows++;
    lp->sum++;
    inc_row_space(lp);

    if(lp->scaling_used)
    {
        /* shift scale */
        for(i=lp->sum; i > lp->rows; i--)
            /*lp->scale[i]=lp->scale[i-1];*/
            mpq_set(lp->scale[i], lp->scale[i-1]);
        /*p->scale[lp->rows]=1;*/
        mpq_set_d(lp->scale[i], 1.0);
    }

    if(lp->names_used)
        sprintf(lp->row_name[lp->rows], "r_%d", lp->rows);

    if(lp->scaling_used && lp->columns_scaled)
        for(i = 1; i <= lp->columns; i++)
            /*row[i] *= lp->scale[lp->rows+i];*/
            mpq_mul(row[i], row[i], lp->scale[lp->rows+i]);

    if(constr_type==GE)
        lp->ch_sign[lp->rows] = TRUE;
    else
        lp->ch_sign[lp->rows] = FALSE;

    elmnr = 0;
    stcol = 0;
    for(i = 1; i <= lp->columns; i++)
    {
        for(j = stcol; j < lp->col_end[i]; j++)
        {
            newmat[elmnr].row_nr=lp->mat[j].row_nr;
            newmat[elmnr].value=lp->mat[j].value;
            elmnr++;
        }
        if(addtoo[i])
        {
            if(lp->ch_sign[lp->rows]) {
                /*newmat[elmnr].value = -row[i];*/
                mpq_neg(temp, row[i]);
                mpq_set(newmat[elmnr].value, temp);
            }
            else {
                /*newmat[elmnr].value = row[i];*/
                mpq_set(newmat[elmnr].value, row[i]);
            }
            newmat[elmnr].row_nr = lp->rows;
            elmnr++;
        }
        stcol = lp->col_end[i];
        lp->col_end[i] = elmnr;
    }

    memcpy(lp->mat, newmat, lp->non_zeros * sizeof(matrec));

    free(newmat);
    free(addtoo);

    for(i = lp->sum; i > lp->rows; i--)
    {
        lp->orig_upbo[i]   = lp->orig_upbo[i - 1];
        lp->orig_lowbo[i]  = lp->orig_lowbo[i - 1];
        lp->basis[i]       = lp->basis[i - 1];
        lp->lower[i]       = lp->lower[i - 1];
        lp->must_be_int[i] = lp->must_be_int[i - 1];
    }

    /* changed from i <= lp->rows to i < lp->rows, MB */
    for(i = 1 ; i < lp->rows; i++)
        if(lp->bas[i] >= lp->rows)
            lp->bas[i]++;

    if(constr_type == LE || constr_type == GE)
    {
        lp->orig_upbo[lp->rows] = lp->infinite;
    }
    else if(constr_type == EQ)
    {
        lp->orig_upbo[lp->rows] = 0;
    }
    else
    {
        fprintf(stderr, "Wrong constraint type\n");
        exit(EXIT_FAILURE);
    }

    lp->orig_lowbo[lp->rows] = 0;

    if(constr_type == GE && rh != 0) {
        /*lp->orig_rh[lp->rows] =- rh;*/
        mpq_neg(temp, rh);
        mpq_set(lp->orig_rh[lp->rows], temp);
    }
    else {
        /*lp->orig_rh[lp->rows] = rh;*/
        mpq_set(lp->orig_rh[lp->rows], rh);
    }

    lp->row_end_valid = FALSE;

    lp->bas[lp->rows] = lp->rows;
    lp->basis[lp->rows] = TRUE;
    lp->lower[lp->rows] = TRUE;
    lp->eta_valid = FALSE;
}

void str_add_constraint(lprec *lp,
                        char *row_string,
                        short constr_type,
                        REAL rh)
{
    int  i;
    double *aRow;
    char *p, *newp;
    CALLOC(aRow, lp->columns + 1);
    REAL aRow_REAL[lp->columns+1];

    p = row_string;

    for(i = 1; i <= lp->columns; i++)
    {
        aRow[i] = (double) strtod(p, &newp);
        mpq_init(aRow_REAL[i]);
        mpq_set_d(aRow_REAL[i], aRow[i]);
        if(p==newp)
            error("Bad string in str_add_constr");
        else
            p=newp;
    }
    add_constraint(lp, aRow_REAL, constr_type, rh);
    for(i = 1; i <= lp->columns; i++)
        mpq_clear(aRow_REAL[i]);
    free(aRow);
    free(aRow_REAL);
}

void del_constraint(lprec *lp, int del_row)
{
    int i, j;
    unsigned elmnr;
    int startcol;

    if(del_row<1 || del_row>lp->rows)
    {
        fprintf(stderr, "There is no constraint nr. %d\n", del_row);
        exit(EXIT_FAILURE);
    }

    elmnr=0;
    startcol=0;

    for(i = 1; i <= lp->columns; i++)
    {
        for(j=startcol; j < lp->col_end[i]; j++)
        {
            if(lp->mat[j].row_nr!=del_row)
            {
                lp->mat[elmnr]=lp->mat[j];
                if(lp->mat[elmnr].row_nr > del_row)
                    lp->mat[elmnr].row_nr--;
                elmnr++;
            }
            else
                lp->non_zeros--;
        }
        startcol=lp->col_end[i];
        lp->col_end[i]=elmnr;
    }
    for(i = del_row; i < lp->rows; i++)
    {
        lp->orig_rh[i] = lp->orig_rh[i + 1];
        lp->ch_sign[i] = lp->ch_sign[i + 1];
        lp->bas[i] = lp->bas[i + 1];
        if(lp->names_used)
            strcpy(lp->row_name[i], lp->row_name[i + 1]);
    }
    for(i = 1; i < lp->rows; i++)
        if(lp->bas[i] >  del_row)
            lp->bas[i]--;

    for(i=del_row; i < lp->sum; i++)
    {
        lp->lower[i]=lp->lower[i + 1];
        lp->basis[i]=lp->basis[i + 1];
        mpq_set(lp->orig_upbo[i], lp->orig_upbo[i+1]);//lp->orig_upbo[i]=lp->orig_upbo[i + 1];
        mpq_set(lp->orig_lowbo[i], lp->orig_lowbo[i+1]);//lp->orig_lowbo[i]=lp->orig_lowbo[i + 1];
        lp->must_be_int[i]=lp->must_be_int[i + 1];
        if(lp->scaling_used)
            lp->scale[i]=lp->scale[i + 1];
    }

    lp->rows--;
    lp->sum--;

    lp->row_end_valid = FALSE;
    lp->eta_valid     = FALSE;
    lp->basis_valid   = FALSE;
}

void add_lag_con(lprec *lp, REAL *row, short con_type, REAL rhs)
{
    int i;
    /*REAL sign = 1;*/
    REAL sign;
    mpq_init(sign);
    mpq_set_ui(sign,1,1); //Sets sign to 1/1 == 1
    if(con_type == LE || con_type == EQ){}//we don't need to set sign to 1, since it is already 1.
        /*sign = 1;*/
    else if(con_type == GE)
        /*sign = -1;*/
        mpq_set_si(sign,-1,1); //sign = -1/1 == -1
    else
        error("con_type not implemented\n");

    lp->nr_lagrange++;
    if(lp->nr_lagrange==1)
    {
        CALLOC(lp->lag_row, lp->nr_lagrange);
        CALLOC(lp->lag_rhs, lp->nr_lagrange);
        CALLOC(lp->lambda, lp->nr_lagrange);
        //VS We loop over nr_lagrange and initialize that many mpq_ts
        for(i = 1; i < lp->nr_lagrange; i++){
            mpq_init(*lp->lag_rhs[i]);
            mpq_init(*lp->lambda[i]);
        }
        CALLOC(lp->lag_con_type, lp->nr_lagrange);
    }
    else
    {
        REALLOC(lp->lag_row, lp->nr_lagrange);
        REALLOC(lp->lag_rhs, lp->nr_lagrange);
        REALLOC(lp->lambda, lp->nr_lagrange);
        //VS We loop over nr_lagrange and initialize that many mpq_ts
        for(i = 1; i < lp->nr_lagrange; i++){
            mpq_init(lp->lag_rhs[i]);
            mpq_init(lp->lambda[i]);
        }
        REALLOC(lp->lag_con_type, lp->nr_lagrange);
    }
    CALLOC(lp->lag_row[lp->nr_lagrange-1], lp->columns+1);

    /*lp->lag_rhs[lp->nr_lagrange-1]=rhs * sign;*/
    mpq_mul(lp->lag_rhs[lp->nr_lagrange-1], rhs, sign);

    for( i=1; i <= lp->columns; i++){
        /*lp->lag_row[lp->nr_lagrange-1][i]=row[i] * sign;*/
        mpq_init(lp->lag_row[lp->nr_lagrange-1][i]);
        mpq_mul(lp->lag_row[lp->nr_lagrange-1][i], row[i], sign);
    }
    /*lp->lambda[lp->nr_lagrange-1]=0;*/
    mpq_init(lp->lambda[lp->nr_lagrange-1]);
    mpq_set_ui(lp->lambda[lp->nr_lagrange-1], 0, 1); //VS set to 0/1 == 0
    lp->lag_con_type[lp->nr_lagrange-1]=(con_type == EQ);
}

void str_add_lag_con(lprec *lp, char *row, short con_type, REAL rhs)
{
    int  i;
    double *a_row;
    char *p, *new_p;
    REAL *a_row_REAL;
    CALLOC(a_row, lp->columns + 1);
    CALLOC(a_row_REAL, lp->columns +1);
    p = row;

    for(i = 1; i <= lp->columns; i++)
    {
        a_row[i] = (double) strtod(p, &new_p);
        mpq_init(a_row_REAL[i]);
        mpq_set_d(a_row_REAL[i], a_row[i]);
        if(p==new_p)
            error("Bad string in str_add_lag_con");
        else
            p=new_p;
    }
    add_lag_con(lp, a_row_REAL, con_type, rhs);
    free(a_row);

    for(i = 0; i <= lp->columns; i++)
        mpq_clear(a_row_REAL[i]);
    free(a_row_REAL);
}


void add_column(lprec *lp, REAL *column)
{
    int i, elmnr;

    /* if the column has only one entry, this should be handled as a bound, but
       this currently is not the case */

    lp->columns++;
    lp->sum++;
    inc_col_space(lp);
    inc_mat_space(lp, lp->rows + 1);

    if(lp->scaling_used)
    {
        for(i = 0; i <= lp->rows; i++) {
            /*column[i]*=lp->scale[i];*/
            mpq_init(column[i]);
            mpq_mul(column[i], column[i], lp->scale[i]);
        }
        mpq_set_ui(lp->scale[lp->sum],1,1); //VS changed lp->scale[lp->sum] = 1;
    }

    elmnr = lp->col_end[lp->columns - 1];
    for(i = 0 ; i <= lp->rows ; i++)
        if(column[i] != 0)
        {
            lp->mat[elmnr].row_nr = i;
            if(lp->ch_sign[i]) {
                /*lp->mat[elmnr].value = -column[i];*/
                mpq_init(lp->mat[elmnr].value);
                mpq_neg(lp->mat[elmnr].value, column[i]);
            }
            else {
                /*lp->mat[elmnr].value = column[i];*/
                mpq_init(lp->mat[elmnr].value);
                mpq_set(lp->mat[elmnr].value, column[i]);
            }
            lp->non_zeros++;
            elmnr++;
        }
    lp->col_end[lp->columns] = elmnr;
    mpq_set_ui(lp->orig_lowbo[lp->sum], 0, 1);//lp->orig_lowbo[lp->sum] = 0;
    mpq_set(lp->orig_upbo[lp->sum], lp->infinite);//lp->orig_upbo[lp->sum] = lp->infinite;
    lp->lower[lp->sum] = TRUE;
    lp->basis[lp->sum] = FALSE;
    lp->must_be_int[lp->sum] = FALSE;
    if(lp->names_used)
        sprintf(lp->col_name[lp->columns], "var_%d", lp->columns);

    lp->row_end_valid = FALSE;
}

void str_add_column(lprec *lp, char *col_string)
{
    int  i;
    double *aCol;
    REAL *aCol_REAL;
    char *p, *newp;
    CALLOC(aCol, lp->rows + 1);
    CALLOC(aCol_REAL, lp->rows +1);
    p = col_string;

    for(i = 0; i <= lp->rows; i++)
    {
        aCol[i] = (double) strtod(p, &newp);
        mpq_init(aCol_REAL[i]);
        mpq_set_d(aCol_REAL[i], aCol[i]);
        if(p==newp)
            error("Bad string in str_add_column");
        else
            p=newp;
    }
    add_column(lp, aCol_REAL);
    for(i = 0; i <= lp->rows; i++)
        mpq_clear(aCol_REAL[i]);
    free(aCol);
    free(aCol_REAL);
}

void del_column(lprec *lp, int column)
{
    int i, j, from_elm, to_elm, elm_in_col;
    if(column > lp->columns || column < 1)
        error("Column out of range in del_column");
    for(i = 1; i <= lp->rows; i++)
    {
        if(lp->bas[i]==lp->rows+column)
            lp->basis_valid=FALSE;
        else if(lp->bas[i] > lp->rows+column)
            lp->bas[i]--;
    }
    for(i = lp->rows+column; i < lp->sum; i++)
    {
        if(lp->names_used)
            strcpy(lp->col_name[i-lp->rows], lp->col_name[i-lp->rows+1]);
        lp->must_be_int[i]=lp->must_be_int[i + 1];
        mpq_set(lp->orig_upbo[i],lp->orig_upbo[i + 1]);
        mpq_set(lp->orig_lowbo[i],lp->orig_lowbo[i + 1]);//lp->orig_lowbo[i]=lp->orig_lowbo[i + 1];
        mpq_set(lp->upbo[i],lp->upbo[i + 1]);//lp->upbo[i]=lp->upbo[i + 1];
        mpq_set(lp->lowbo[i],lp->lowbo[i + 1]);//lp->lowbo[i]=lp->lowbo[i + 1];
        lp->basis[i]=lp->basis[i + 1];
        lp->lower[i]=lp->lower[i + 1];
        if(lp->scaling_used)
            mpq_set(*lp->scale[i],*lp->scale[i + 1]);
    }
    for(i = 0; i < lp->nr_lagrange; i++)
        for(j = column; j <= lp->columns; j++)
            mpq_set(*lp->lag_row[i][j],*lp->lag_row[i][j+1]);
    to_elm=lp->col_end[column-1];
    from_elm=lp->col_end[column];
    elm_in_col=from_elm-to_elm;
    for(i = from_elm; i < lp->non_zeros; i++)
    {
        lp->mat[to_elm] = lp->mat[i];
        to_elm++;
    }
    for(i = column; i < lp->columns; i++)
        lp->col_end[i]=lp->col_end[i + 1]-elm_in_col;
    lp->non_zeros -= elm_in_col;
    lp->row_end_valid=FALSE;
    lp->eta_valid=FALSE;

    lp->sum--;
    lp->columns--;
}

void set_upbo(lprec *lp, int column, REAL value)
{
    REAL temp;
    mpq_init(*temp);
    if(column > lp->columns || column < 1)
        error("Column out of range");
    if(lp->scaling_used)
        mpq_div(*temp, *value, *lp->scale[lp->rows + column]);//value /= lp->scale[lp->rows + column];
    if(mpq_cmp(*temp, *lp->orig_lowbo[lp->rows + column]) < 0)//value < lp->orig_lowbo[lp->rows + column])
        error("Upperbound must be >= lowerbound");
    lp->eta_valid = FALSE;
    mpq_set(*lp->orig_upbo[lp->rows+column], *temp);
    mpq_clear(*temp);
}

void set_lowbo(lprec *lp, int column, REAL value)
{
    REAL temp;
    mpq_init(*temp);
    if(column > lp->columns || column < 1)
        error("Column out of range");
    if(lp->scaling_used)
        mpq_div(*temp, *value, *lp->scale[lp->rows + column]);
    if(mpq_cmp(*temp, *lp->orig_upbo[lp->rows + column]))//value > lp->orig_upbo[lp->rows + column])
        error("Upperbound must be >= lowerbound");
    /*
      if(value < 0)
      error("Lower bound cannot be < 0");
      */
    lp->eta_valid = FALSE;
    mpq_set(*lp->orig_lowbo[lp->rows + column], *temp);//lp->orig_lowbo[lp->rows + column] = value;
}

void set_int(lprec *lp, int column, short must_be_int)
{
    if(column > lp->columns || column < 1)
        error("Column out of range");
    lp->must_be_int[lp->rows+column]=must_be_int;
    if(must_be_int == TRUE)
    if(lp->columns_scaled)
        unscale_columns(lp);
}

void set_rh(lprec *lp, int row, REAL value)
{
    REAL temp;
    mpq_init(*temp);
    mpq_set(*temp, *value);
    if(row > lp->rows || row < 0)
        error("Row out of Range");

    if ((row == 0) && (!lp->maximise))  /* setting of RHS of OF IS meaningful */
        mpq_neg(*temp, *value);//value = -value;
    if(lp->scaling_used) {
        if(lp->ch_sign[row]) {
            //lp->orig_rh[row] = -value * lp->scale[row];
            mpq_neg(*temp, *value);
            mpq_mul(*lp->orig_rh[row], *temp, *lp->scale[row]);
        }
        else
            mpq_mul(*lp->orig_rh[row], *temp, *lp->scale[row]);//lp->orig_rh[row] = value * lp->scale[row];
    }
    else
    if(lp->ch_sign[row]) {
        //lp->orig_rh[row] = -value;
        mpq_neg(*temp, *value);
        mpq_set(*lp->orig_rh[row], *temp);
    }
    else
        mpq_set(*lp->orig_rh[row], *temp);//lp->orig_rh[row] = value;
    lp->eta_valid = FALSE;
    mpq_clear(*temp);
}

void set_rh_vec(lprec *lp, REAL *rh)
{
    int i;

    if(lp->scaling_used) {
        for(i = 1; i <= lp->rows; i++) {
            if (lp->ch_sign[i]) {
                REAL temp;
                mpq_init(*temp);
                //lp->orig_rh[i] = -rh[i] * lp->scale[i];
                mpq_neg(*temp, *rh[i]);
                mpq_mul(*lp->orig_rh[i], *temp, *lp->scale[i]);
                mpq_clear(*temp);
            }
            else
                mpq_mul(*lp->orig_rh[i], *rh[i], *lp->scale[i]);//lp->orig_rh[i] = rh[i] * lp->scale[i];
        }
    }
    else
        for(i=1; i <= lp->rows; i++)
            if(lp->ch_sign[i])
                mpq_neg(*lp->orig_rh[i], *rh[i]);//lp->orig_rh[i]=-rh[i];
            else
                mpq_set(*lp->orig_rh[i], *rh[i]);//lp->orig_rh[i]=rh[i];
    lp->eta_valid=FALSE;
}

void str_set_rh_vec(lprec *lp, char *rh_string)
{
    int  i;
    double *newrh;
    REAL *newrh_REAL;
    char *p, *newp;
    CALLOC(newrh, lp->rows + 1);
    CALLOC(newrh_REAL, lp->rows + 1);
    p = rh_string;

    for(i = 1; i <= lp->rows; i++)
    {
        newrh[i] = (double) strtod(p, &newp);
        mpq_init(*newrh_REAL[i]);
        mpq_set_d(*newrh_REAL[i], newrh[i]);
        if(p==newp)
            error("Bad string in str_set_rh_vec");
        else
            p=newp;
    }
    set_rh_vec(lp, newrh_REAL);
    for(i = 1; i <= lp->rows; i++)
    {
        mpq_clear(*newrh_REAL[i]);
    }
    free(newrh);
    free(newrh_REAL);
}


void set_maxim(lprec *lp)
{
    int i;
    if(lp->maximise == FALSE)
    {
        for(i = 0; i < lp->non_zeros; i++)
            if(lp->mat[i].row_nr == 0)
                mpq_neg(*lp->mat[i].value, *lp->mat[i].value);//lp->mat[i].value *= -1;
        lp->eta_valid = FALSE;
        mpq_neg(*lp->orig_rh[0], *lp->orig_rh[0]);//lp->orig_rh[0] *= -1;
    }
    lp->maximise = TRUE;
    lp->ch_sign[0] = TRUE;
}

void set_minim(lprec *lp)
{
    int i;
    if(lp->maximise == TRUE)
    {
        for(i = 0; i < lp->non_zeros; i++)
            if(lp->mat[i].row_nr == 0)
                mpq_neg(*lp->mat[i].value, *lp->mat[i].value);//lp->mat[i].value = -lp->mat[i].value;
        lp->eta_valid = FALSE;
        mpq_neg(*lp->orig_rh[0], *lp->orig_rh[0]);//lp->orig_rh[0] *= -1;
    }
    lp->maximise = FALSE;
    lp->ch_sign[0] = FALSE;
}

void set_constr_type(lprec *lp, int row, short con_type)
{
    int i;
    if(row > lp->rows || row < 1)
        error("Row out of Range");
    if(con_type==EQ)
    {
        mpq_set_ui(*lp->orig_upbo[row], 0, 1);//lp->orig_upbo[row]=0;
        lp->basis_valid=FALSE;
        if(lp->ch_sign[row])
        {
            for(i = 0; i < lp->non_zeros; i++)
                if(lp->mat[i].row_nr==row)
                    mpq_neg(*lp->mat[i].value, *lp->mat[i].value);//lp->mat[i].value*=-1;
            lp->eta_valid=FALSE;
            lp->ch_sign[row]=FALSE;
            if(lp->orig_rh[row]!=0)
                mpq_neg(*lp->orig_rh[row], *lp->orig_rh[row]);//lp->orig_rh[row]*=-1;
        }
    }
    else if(con_type==LE)
    {
        mpq_set(*lp->orig_upbo[row],*lp->infinite);
        lp->basis_valid=FALSE;
        if(lp->ch_sign[row])
        {
            for(i = 0; i < lp->non_zeros; i++)
                if(lp->mat[i].row_nr==row)
                    mpq_neg(*lp->mat[i].value, *lp->mat[i].value);//lp->mat[i].value*=-1;
            lp->eta_valid=FALSE;
            lp->ch_sign[row]=FALSE;
            if(lp->orig_rh[row]!=0)
                mpq_neg(*lp->orig_rh[row], *lp->orig_rh[row]);//lp->orig_rh[row]*=-1;
        }
    }
    else if(con_type==GE)
    {
        mpq_set(*lp->orig_upbo[row],*lp->infinite);
        lp->basis_valid=FALSE;
        if(!lp->ch_sign[row])
        {
            for(i = 0; i < lp->non_zeros; i++)
                if(lp->mat[i].row_nr==row)
                    mpq_neg(*lp->mat[i].value, *lp->mat[i].value);//lp->mat[i].value*=-1;
            lp->eta_valid=FALSE;
            lp->ch_sign[row]=TRUE;
            if(lp->orig_rh[row]!=0)
                mpq_neg(*lp->orig_rh[row], *lp->orig_rh[row]);//lp->orig_rh[row]*=-1;
        }
    }
    else
        error("Constraint type not (yet) implemented");
}

REAL mat_elm(lprec *lp, int row, int column)
{
    REAL value;
    int elmnr;
    if(row < 0 || row > lp->rows)
        error("Row out of range in mat_elm");
    if(column < 1 || column > lp->columns)
        error("Column out of range in mat_elm");
    mpq_init(*value);//value=0;
    elmnr=lp->col_end[column-1];
    while(lp->mat[elmnr].row_nr != row && elmnr < lp->col_end[column])
        elmnr++;
    if(elmnr != lp->col_end[column])
    {
        mpq_set(*value, *lp->mat[elmnr].value);//value = lp->mat[elmnr].value;
        if(lp->ch_sign[row])
            mpq_neg(*value, *value);//value = -value;
        if(lp->scaling_used) {
            //value /= lp->scale[row] * lp->scale[lp->rows + column];
            REAL temp;
            mpq_init(*temp);
            mpq_mul(*temp, *lp->scale[row], *lp->scale[lp->rows + column]);
            mpq_div(*value, *value, *temp);
            mpq_clear(*temp);
        }

    }
    return(value);
}


void get_row(lprec *lp, int row_nr, REAL *row)
{
    /* VS - get_row expects the array at *row has been allocated memory, but not initialized its mpqs (REALs) */
    int i, j;

    if(row_nr <0 || row_nr > lp->rows)
        error("Row nr. out of range in get_row");
    for(i = 1; i <= lp->columns; i++)
    {
        mpq_init(*row[i]);//row[i]=0;
        for(j=lp->col_end[i-1]; j < lp->col_end[i]; j++)
            if(lp->mat[j].row_nr==row_nr)
                mpq_set(*row[i], *lp->mat[j].value);//row[i]=lp->mat[j].value;
        if(lp->scaling_used) {
            //row[i] /= lp->scale[lp->rows + i] * lp->scale[row_nr];
            REAL temp;
            mpq_init(*temp);
            mpq_mul(*temp, *lp->scale[lp->rows + i], *lp->scale[row_nr]);
            mpq_div(*row[i], *row[i], *temp);
        }
    }
    if(lp->ch_sign[row_nr])
        for(i=0; i <= lp->columns; i++)
            if(mpq_sgn(*row[i]) != 0)//row[i]!=0)
                mpq_neg(*row[i], *row[i]);//row[i] = -row[i];
}

void get_column(lprec *lp, int col_nr, REAL *column)
{
    int i;

    if(col_nr < 1 || col_nr > lp->columns)
        error("Col. nr. out of range in get_column");
    for(i = 0; i <= lp->rows; i++)
        mpq_set_ui(*column[i], 0, 1);//column[i]=0;
    for(i = lp->col_end[col_nr-1]; i < lp->col_end[col_nr]; i++)
        mpq_set(*column[lp->mat[i].row_nr], *lp->mat[i].value);//column[lp->mat[i].row_nr] = lp->mat[i].value;
    for(i = 0; i <= lp->rows; i++)
        if(mpq_sgn(*column[i]) != 0)//column[i] != 0)
        {
            if(lp->ch_sign[i])
                mpq_neg(*column[i], *column[i]);//column[i] *= -1;
            if(lp->scaling_used) {
                //column[i] /= (lp->scale[i] * lp->scale[lp->rows + col_nr]);
                REAL temp;
                mpq_init(*temp);
                mpq_mul(*temp, *lp->scale[i], *lp->scale[lp->rows + col_nr]);
                mpq_div(*column[i], *column[i], *temp);
                mpq_clear(*temp);
            }
        }
}

void get_reduced_costs(lprec *lp, REAL *rc)
{
    int varnr, i, j;
    REAL f;

    if(!lp->basis_valid)
        error("Not a valid basis in get_reduced_costs");

    if(!lp->eta_valid)
        invert(lp);

    for(i = 1; i <= lp->sum; i++)
        mpq_set_ui(*rc[i], 0, 1);//rc[i] = 0;
    mpq_set_ui(*rc[0], 1, 1);//rc[0] = 1;

    lp_solve_btran(lp, rc);

    for(i = 1; i <= lp->columns; i++)
    {
        varnr = lp->rows + i;
        if(!lp->basis[varnr])
        if(mpq_sgn(*lp->upbo[varnr]) > 0)//lp->upbo[varnr] > 0)
        {
            mpq_init(*f);//f = 0;
            REAL temp;
            mpq_init(*temp);
            for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++){
                //f += rc[lp->mat[j].row_nr] * lp->mat[j].value;
                mpq_mul(*temp, *rc[lp->mat[j].row_nr], *lp->mat[j].value);
                mpq_add(*f, *f, *temp);
            }
            mpq_clear(*f);
            mpq_clear(*temp); //TODO: Allocating and deallocating memory for f and temp is slow - maybe just allocate for f and temp outside of loop?
            mpq_set(*rc[varnr], *f);//rc[varnr] = f;
        }
    }
    /* VS - I don't think we need to round here, since we're using REAL/Rationals */
    /*for(i = 1; i <= lp->sum; i++)
        my_round(rc[i], lp->epsd);
    */
}

short is_feasible(lprec *lp, REAL *values)
{
    int i, elmnr, j;
    REAL *this_rhs;
    REAL dist;
    REAL temp;
    mpq_init(*temp);


    if(lp->scaling_used)
    {
        for(i = lp->rows + 1; i <= lp->sum; i++) {
            /*if(   values[i - lp->rows] < lp->orig_lowbo[i] * lp->scale[i]
                  || values[i - lp->rows] > lp->orig_upbo[i]  * lp->scale[i])*/

            mpq_mul(*temp, *lp->orig_lowbo[i], *lp->scale[i]);
            if (mpq_cmp(*values[i - lp->rows], *temp) < 0) {
                mpq_clear(*temp);
                return (FALSE);
            }

            mpq_mul(*temp, *lp->orig_upbo[i], *lp->scale[i]);
            if (mpq_cmp(*values[i - lp->rows], *temp) > 0) {
                mpq_clear(*temp);
                return (FALSE);
            }
        }
    }
    else
    {
        for(i = lp->rows + 1; i <= lp->sum; i++) {
            /*if(   values[i - lp->rows] < lp->orig_lowbo[i]
                  || values[i - lp->rows] > lp->orig_upbo[i])*/
            if (mpq_cmp(*values[i - lp->rows], *lp->orig_lowbo[i]) < 0
                || mpq_cmp(*values[i - lp->rows], *lp->orig_upbo[i]) > 0) {
                mpq_clear(*temp);
                return (FALSE);
            }
        }
    }
    CALLOC(this_rhs, lp->rows + 1);
    for(i = 0; i <= lp->rows; i++)
        mpq_init(*this_rhs[i]);

    for(i = 1; i <= lp->columns; i++)
        for(elmnr = lp->col_end[i - 1]; elmnr < lp->col_end[i]; elmnr++) {
            //this_rhs[lp->mat[elmnr].row_nr] += lp->mat[elmnr].value * values[i];
            mpq_mul(*temp, *lp->mat[elmnr].value, *values[i]);
            mpq_add(*this_rhs[lp->mat[elmnr].row_nr], *this_rhs[lp->mat[elmnr].row_nr], *temp);
        }
    for(i = 1; i <= lp->rows; i++)
    {
        //dist = lp->orig_rh[i] - this_rhs[i];
        mpq_init(*dist);
        mpq_sub(*dist, *lp->orig_rh[i], *this_rhs[i]);
        //my_round(dist, 0.001) /* ugly constant, MB */ VS - We don't need to round since we're using rationals.
        if((mpq_sgn(*lp->orig_upbo[i]) == 0 && mpq_sgn(*dist) != 0) || mpq_sgn(*dist) < 0)//if((lp->orig_upbo[i] == 0 && dist != 0) || dist < 0)
        {
            for(j = 0; j <= lp->rows; j++){
                mpq_clear(*this_rhs[j]);
            }
            free(this_rhs);
            mpq_clear(*temp);
            mpq_clear(*dist);

            return(FALSE);
        }
    }
    for(j = 0; j <= lp->rows; j++){
        mpq_clear(*this_rhs[j]);
    }
    free(this_rhs);
    mpq_clear(*temp);
    mpq_clear(*dist);
    return(TRUE);
}

/* fixed by Enrico Faggiolo */
short column_in_lp(lprec *lp, REAL *testcolumn)
{
    int i, j;
    int nz, ident;
    REAL value;
    REAL temp;
    mpq_init(*value);
    mpq_init(*temp);

    for(nz = 0, i = 0; i <= lp->rows; i++) {
        //if (my_abs(testcolumn[i]) > lp->epsel) nz++;
        mpq_abs(*temp, *testcolumn[i]);
        if(mpq_cmp(*temp, *lp->epsel) > 0) nz++;
    }

    if(lp->scaling_used)
        for(i = 1; i <= lp->columns; i++)
        {
            ident = nz;
            for(j = lp->col_end[i - 1]; j < lp->col_end[i]; j++)
            {
                mpq_set(*value, *lp->mat[j].value);//value = lp->mat[j].value;
                if(lp->ch_sign[lp->mat[j].row_nr])
                    mpq_neg(*value, *value);//value = -value;
                mpq_div(*value, *value, *lp->scale[lp->rows+i]);//value /= lp->scale[lp->rows+i];
                mpq_div(*value, *value, *lp->scale[lp->mat[j].row_nr]);//value /= lp->scale[lp->mat[j].row_nr];
                mpq_sub(*value, *value, *testcolumn[lp->mat[j].row_nr]);//value -= testcolumn[lp->mat[j].row_nr];

                mpq_abs(*temp, *value);
                if(mpq_cmp(*temp, *lp->epsel) > 0)//if(my_abs(value) > lp->epsel)
                    break;
                ident--;
                if(ident == 0) {
                    mpq_clear(*value);
                    mpq_clear(*temp);
                    return (TRUE);
                }
            }
        }
    else
        for(i = 1; i <= lp->columns; i++)
        {
            ident = nz;
            for(j = lp->col_end[i-1]; j < lp->col_end[i]; j++)
            {
                mpq_set(*value, *lp->mat[j].value);//value = lp->mat[j].value;
                if(lp->ch_sign[lp->mat[j].row_nr])
                    mpq_neg(*value, *value);//value = -value;
                mpq_sub(*value, *value, *testcolumn[lp->mat[j].row_nr]);//value -= testcolumn[lp->mat[j].row_nr];
                mpq_abs(*temp, *value);
                if(mpq_cmp(*temp, *lp->epsel) > 0)//if(my_abs(value) > lp->epsel)
                    break;
                ident--;
                if(ident == 0) {
                    mpq_clear(*value);
                    mpq_clear(*temp);
                    return (TRUE);
                }
            }
        }
    mpq_clear(*value);
    mpq_clear(*temp);
    return(FALSE);
}

void print_lp(lprec *lp)
{
    int i, j;
    REAL *fatmat;
    REAL temp;
    mpq_init(*temp);
    CALLOC(fatmat, (lp->rows + 1) * lp->columns);

    for(i = 0; i < ((lp->rows + 1) * lp->columns); i++)
        mpq_init(*fatmat[i]);

    for(i = 1; i <= lp->columns; i++)
        for(j = lp->col_end[i-1]; j < lp->col_end[i]; j++)
            mpq_set(*fatmat[(i - 1) * (lp->rows + 1) + lp->mat[j].row_nr], *lp->mat[j].value);

    printf("problem name: %s\n", lp->lp_name);
    printf("          ");
    for(j = 1; j <= lp->columns; j++)
        if(lp->names_used)
            printf("%8s ", lp->col_name[j]);
        else
            printf("Var[%3d] ", j);
    if(lp->maximise)
    {
        printf("\nMaximise  ");
        for(j = 0; j < lp->columns; j++) {
            mpq_neg(*temp, *fatmat[j * (lp->rows +1)]);
            mpq_out_str(stdout, 10, *temp);//printf("% 8.2f ", -fatmat[j * (lp->rows + 1)]);
        }
    }
    else
    {
        printf("\nMinimize  ");
        for(j = 0; j < lp->columns; j++)
            mpq_out_str(stdout, 10, *fatmat[j * (lp->rows + 1)]);//printf("% 8.2f ", fatmat[j * (lp->rows + 1)]);
    }
    printf("\n");
    for(i = 1; i <= lp->rows; i++)
    {
        if(lp->names_used)
            printf("%9s ", lp->row_name[i]);
        else
            printf("Row[%3d]  ", i);
        for(j = 0; j < lp->columns; j++)
            if(lp->ch_sign[i] && mpq_sgn(*fatmat[j*(lp->rows+1)+i]) != 0) {
                //printf("% 8.2f ",-fatmat[j*(lp->rows+1)+i]);
                mpq_neg(*temp,*fatmat[j*(lp->rows+1)+i] );
                mpq_out_str(stdout, 10, *temp);
            }
            else
                mpq_out_str(stdout, 10, *fatmat[j*(lp->rows+1)+i]);//printf("% 8.2f ", fatmat[j*(lp->rows+1)+i]);
        if(mpq_sgn(*lp->orig_upbo[i]) != 0){//lp->orig_upbo[i] != 0) {
            if(lp->ch_sign[i])
                printf(">= ");
            else
                printf("<= ");
        }
        else
            printf(" = ");
        if(lp->ch_sign[i]) {
            //printf("% 8.2f", -lp->orig_rh[i]);
            mpq_neg(*temp, *lp->orig_rh[i]);
            mpq_out_str(stdout, 10, *temp);
        }
        else
            mpq_out_str(stdout, 10, *lp->orig_rh[i]);//printf("% 8.2f", lp->orig_rh[i]);
        if(mpq_sgn(*lp->orig_lowbo[i]) != 0){//lp->orig_lowbo[i] != 0) {
            printf("  %s=", (lp->ch_sign[i]) ? "lowbo" : "upbo");//printf("  %s=%8.2f", (lp->ch_sign[i]) ? "lowbo" : "upbo",
                   //lp->orig_lowbo[i]);
            mpq_out_str(stdout, 10, *lp->orig_lowbo[i]);
        }
        if(mpq_cmp(*lp->orig_upbo[i], *lp->infinite) != 0 && mpq_sgn(*lp->orig_upbo[i]) != 0){//(lp->orig_upbo[i]!=lp->infinite) && (lp->orig_upbo[i]!=0.0)) {
            printf("  %s=", (lp->ch_sign[i]) ? "upbo" : "lowbo");//printf("  %s=%8.2f", (lp->ch_sign[i]) ? "upbo" : "lowbo",
                   //lp->orig_upbo[i]);
            mpq_out_str(stdout, 10, *lp->orig_upbo[i]);
        }
        printf("\n");
    }
    printf("Type      ");
    for(i = 1; i <= lp->columns; i++)
        if(lp->must_be_int[lp->rows+i]==TRUE)
            printf("     Int ");
        else
            printf("    Real ");
    printf("\nupbo      ");
    for(i = 1; i <= lp->columns; i++)
        if(mpq_cmp(*lp->orig_upbo[lp->rows+i], *lp->infinite) == 0)//lp->orig_upbo[lp->rows+i]==lp->infinite)
            printf("     Inf ");
        else
            mpq_out_str(stdout, 10, *lp->orig_upbo[lp->rows+i]);//printf("% 8.2f ", lp->orig_upbo[lp->rows+i]);
    printf("\nlowbo     ");
    for(i = 1; i <= lp->columns; i++)
        mpq_out_str(stdout, 10, *lp->orig_lowbo[lp->rows+i]);//printf("% 8.2f ", lp->orig_lowbo[lp->rows+i]);
    printf("\n");
    for(i = 0; i < lp->nr_lagrange; i++)
    {
        printf("lag[%3d]  ", i);
        for(j = 1; j <= lp->columns; j++)
            mpq_out_str(stdout, 10, *lp->lag_row[i][j]);//printf("% 8.2f ", lp->lag_row[i][j]);
        if(mpq_cmp(*lp->orig_upbo[i], *lp->infinite) == 0){//lp->orig_upbo[i]==lp->infinite) {
            if(lp->lag_con_type[i] == GE)
                printf(">= ");
            else if(lp->lag_con_type[i] == LE)
                printf("<= ");
            else if(lp->lag_con_type[i] == EQ)
                printf(" = ");
            mpq_out_str(stdout, 10, *lp->lag_rhs[i]);//printf("% 8.2f\n", lp->lag_rhs[i]);
        }
    }
    for(i = 0; i < (lp->rows + 1) * lp->columns; i++)
        mpq_clear(*fatmat[i]);
    mpq_clear(*temp);

    free(fatmat);
}

void set_row_name(lprec *lp, int row, nstring new_name)
{
    int i;
    hashelem *hp;

    if(!lp->names_used)
    {
        CALLOC(lp->row_name, lp->rows_alloc + 1);
        CALLOC(lp->col_name, lp->columns_alloc + 1);
        lp->names_used = TRUE;
        for(i = 0; i <= lp->rows; i++)
            sprintf(lp->row_name[i], "r_%d", i);
        for(i = 1; i <= lp->columns; i++)
            sprintf(lp->col_name[i], "var_%d", i);
    }
    strcpy(lp->row_name[row], new_name);
    hp = puthash(lp->row_name[row], lp->rowname_hashtab);
    hp->index = row;
}

void set_col_name(lprec *lp, int column, nstring new_name)
{
    int i;
    hashelem *hp;

    if(!lp->names_used)
    {
        CALLOC(lp->row_name, lp->rows_alloc + 1);
        CALLOC(lp->col_name, lp->columns_alloc + 1);
        lp->names_used = TRUE;
        for(i = 0; i <= lp->rows; i++)
            sprintf(lp->row_name[i], "r_%d", i);
        for(i = 1; i <= lp->columns; i++)
            sprintf(lp->col_name[i], "var_%d", i);
    }
    strcpy(lp->col_name[column], new_name);
    hp = puthash(lp->col_name[column], lp->colname_hashtab);
    hp->index = column;
}

static REAL minmax_to_scale(REAL min, REAL max)
{
    REAL scale;
    mpq_init(*scale);
    /* should do something sensible when min or max is 0, MB */
    if(mpq_sgn(*min) == 0 || mpq_sgn(*max) == 0) {//(min == 0) || (max == 0))
        mpq_set_ui(*scale, 1, 1);
        return (scale);
    }

    //scale = 1 / pow(10, (log10(min) + log10(max)) / 2);
    /*VS - no exact algorithm exists for calculating logarithm, so we have to convert min and max to doubles
     * We could implement a Taylor series approximation with Rationals,
     * but it still would be approximate - better to use a faster approximation?
     */
    mpq_mul(scale, min, max); //we use the logarithm identity log(x) + log(y) = log(x*y)
    mpq_set_d(scale, (log10(mpq_get_d(scale)))); //have to convert to double here.
    mpq_div_2exp(scale, scale, 1); //divide by 2
    mpq_set_d(scale, pow(10, mpq_get_d(scale))); //also convert to double here. Maybe we don't need to?
    mpq_inv(scale, scale);

    return(scale);
}

void unscale_columns(lprec *lp)
{
    int i, j;

    /* unscale mat */
    for(j = 1; j <= lp->columns; j++)
        for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
            mpq_div(lp->mat[i].value, lp->mat[i].value, lp->scale[lp->rows + j]);//lp->mat[i].value /= lp->scale[lp->rows + j];

    /* unscale bounds as well */
    for(i = lp->rows + 1; i <= lp->sum; i++) /* was < */ /* changed by PN */
    {
        if(mpq_sgn(lp->orig_lowbo[i]) != 0)//lp->orig_lowbo[i] != 0)
            mpq_mul(lp->orig_lowbo[i], lp->orig_lowbo[i], lp->scale[i]);//lp->orig_lowbo[i] *= lp->scale[i];
        if(mpq_cmp(lp->orig_upbo[i], lp->infinite))//lp->orig_upbo[i] != lp->infinite)
            mpq_mul(lp->orig_upbo[i], lp->orig_upbo[i], lp->scale[i]);//lp->orig_upbo[i] *= lp->scale[i];
    }

    for(i=lp->rows+1; i<= lp->sum; i++)
        mpq_set_ui(lp->scale[i], 1, 1);//lp->scale[i]=1;
    lp->columns_scaled=FALSE;
    lp->eta_valid=FALSE;
}

void unscale(lprec *lp)
{
    int i, j;

    if(lp->scaling_used)
    {

        /* unscale mat */
        for(j = 1; j <= lp->columns; j++)
            for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
                mpq_div(lp->mat[i].value, lp->mat[i].value, lp->scale[lp->rows + j]);//lp->mat[i].value /= lp->scale[lp->rows + j];

        /* unscale bounds */
        for(i = lp->rows + 1; i <= lp->sum; i++) /* was < */ /* changed by PN */
        {
            if(mpq_sgn(lp->orig_lowbo[i]) != 0)//lp->orig_lowbo[i] != 0)
                mpq_mul(lp->orig_lowbo[i], lp->orig_lowbo[i], lp->scale[i]);//lp->orig_lowbo[i] *= lp->scale[i];
            if(mpq_cmp(lp->orig_upbo[i], lp->infinite) != 0)//lp->orig_upbo[i] != lp->infinite)
                mpq_mul(lp->orig_upbo[i], lp->orig_upbo[i], lp->scale[i]);//lp->orig_upbo[i] *= lp->scale[i];
        }

        /* unscale the matrix */
        for(j = 1; j <= lp->columns; j++)
            for(i = lp->col_end[j-1]; i < lp->col_end[j]; i++)
                mpq_div(lp->mat[i].value, lp->mat[i].value, lp->scale[lp->mat[i].row_nr]);//lp->mat[i].value /= lp->scale[lp->mat[i].row_nr];

        /* unscale the rhs! */
        for(i = 0; i <= lp->rows; i++)
            mpq_div(lp->orig_rh[i], lp->orig_rh[i], lp->scale[i]);//lp->orig_rh[i] /= lp->scale[i];

        /* and don't forget to unscale the upper and lower bounds ... */
        for(i = 0; i <= lp->rows; i++)
        {
            if(mpq_sgn(lp->orig_lowbo[i]) != 0)//lp->orig_lowbo[i] != 0)
                mpq_div(lp->orig_lowbo[i], lp->orig_lowbo[i], lp->scale[i]);//lp->orig_lowbo[i] /= lp->scale[i];
            if(mpq_cmp(lp->orig_upbo[i], lp->infinite) != 0)//lp->orig_upbo[i] != lp->infinite)
                mpq_div(lp->orig_upbo[i], lp->orig_upbo[i], lp->scale[i]);//lp->orig_upbo[i] /= lp->scale[i];

            //VS - Since this is the last pace we use lp->scale[i] before freeing the memory, we'll clear them here.
            mpq_clear(lp->scale[i]);
        }

        free(lp->scale);
        lp->scaling_used=FALSE;
        lp->eta_valid=FALSE;
    }
}


void auto_scale(lprec *lp)
{
    int i, j, row_nr, IntUsed;
    REAL *row_max;
    REAL *row_min;
    REAL *scalechange;
    REAL absval;

    if(!lp->scaling_used)
    {
        MALLOC(lp->scale, lp->sum_alloc + 1);
        for(i = 0; i <= lp->sum; i++) {
            mpq_init(lp->scale[i]);//lp->scale[i] = 1;
            mpq_set_ui(lp->scale[i], 1, 1);
        }
    }

    MALLOC(row_max, lp->rows + 1);
    MALLOC(row_min, lp->rows + 1);
    MALLOC(scalechange, lp->sum + 1);

    /* initialise min and max values */
    for(i = 0; i <= lp->rows; i++)
    {
        mpq_init(row_max[i]);//row_max[i] = 0;
        mpq_init(row_min[i]);//row_min[i] = lp->infinite;
        mpq_set(row_min[i], lp->infinite);

        mpq_init(scalechange[i]);
    }

    /* calculate min and max absolute values of rows */
    mpq_init(absval);
    for(j = 1; j <= lp->columns; j++)
        for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
        {
            row_nr = lp->mat[i].row_nr;
            mpq_abs(absval, (lp->mat[i].value));//absval = my_abs(lp->mat[i].value);
            if(mpq_sgn(absval) != 0)//absval != 0)
            {
                mpq_set(row_max[row_nr], *(my_mpq_max(row_max[row_nr], absval)));//row_max[row_nr] = my_mpq_max(row_max[row_nr], absval);
                mpq_set(row_min[row_nr], *(my_mpq_min(row_min[row_nr], absval)));//row_min[row_nr] = my_mpq_min(row_min[row_nr], absval);
            }
        }
    /* calculate scale factors for rows */
    for(i = 0; i <= lp->rows; i++)
    {
        mpq_set(scalechange[i], *minmax_to_scale(row_min[i], row_max[i]));//scalechange[i] = minmax_to_scale(row_min[i], row_max[i]);
        mpq_mul(lp->scale[i], lp->scale[i], scalechange[i]);//lp->scale[i] *= scalechange[i];
        /* VS - since we don't need row_max or row_min anymore, we'll deallocate memory for their mpqs. */
        mpq_clear(row_min[i]);
        mpq_clear(row_max[i]);
    }

    /* now actually scale the matrix */
    for(j = 1; j <= lp->columns; j++)
        for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
            mpq_mul(lp->mat[i].value, lp->mat[i].value, scalechange[lp->mat[i].row_nr]);//lp->mat[i].value *= scalechange[lp->mat[i].row_nr];

    /* and scale the rhs and the row bounds (RANGES in MPS!!) */
    for(i = 0; i <= lp->rows; i++)
    {
        mpq_mul(lp->orig_rh[i], *lp->orig_rh[i], *scalechange[i]);//lp->orig_rh[i] *= scalechange[i];

        if((mpq_cmp(lp->orig_upbo[i], lp->infinite) < 0) && (mpq_sgn(lp->orig_upbo[i]) != 0))//(lp->orig_upbo[i] < lp->infinite) && (lp->orig_upbo[i] != 0))
            mpq_mul(lp->orig_upbo[i], lp->orig_upbo[i], scalechange[i]);//lp->orig_upbo[i] *= scalechange[i];

        if(mpq_sgn(lp->orig_lowbo[i]) != 0)//lp->orig_lowbo[i] != 0) /* can this happen? what would it mean? */
            mpq_mul(lp->orig_lowbo[i], lp->orig_lowbo[i], scalechange[i]);//lp->orig_lowbo[i] *= scalechange[i];

    }



    free(row_max);
    free(row_min);

    for(IntUsed = FALSE, i = lp->rows + 1; !IntUsed && i <= lp->sum; i++)
        IntUsed = lp->must_be_int[i];

    if(!IntUsed)
    {
        REAL col_max;
        REAL col_min;
        REAL temp;
        mpq_init(col_max);
        mpq_init(col_min);
        mpq_init(temp);
        /* calculate column scales */
        for(j = 1; j <= lp->columns; j++)
        {
            mpq_set_ui(col_max, 0, 1);//col_max = 0;
            mpq_set(col_min, lp->infinite);//col_min = lp->infinite;
            for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
            {
                if(mpq_sgn(lp->mat[i].value) != 0)//lp->mat[i].value!=0)
                {
                    mpq_abs(temp, lp->mat[i].value);
                    mpq_set(col_max, *(my_mpq_max(col_max, temp)));//col_max = my_max(col_max, my_abs(lp->mat[i].value));
                    mpq_set(col_min, *(my_mpq_min(col_min, temp)));//col_min = my_min(col_min, my_abs(lp->mat[i].value));
                }
            }
            mpq_set(scalechange[lp->rows + j], *(minmax_to_scale(col_min, col_max)));//scalechange[lp->rows + j]  = minmax_to_scale(col_min, col_max);
            mpq_mul(lp->scale[lp->rows + j], lp->scale[lp->rows + j], scalechange[lp->rows + j]);//lp->scale[lp->rows + j] *= scalechange[lp->rows + j];
        }

        /* scale mat */
        for(j = 1; j <= lp->columns; j++)
            for(i = lp->col_end[j - 1]; i < lp->col_end[j]; i++)
                mpq_mul(lp->mat[i].value, lp->mat[i].value, scalechange[lp->rows + j]);//lp->mat[i].value *= scalechange[lp->rows + j];

        /* scale bounds as well */
        for(i = lp->rows + 1; i <= lp->sum; i++) /* was < */ /* changed by PN */
        {
            if(mpq_sgn(lp->orig_lowbo[i]) != 0)//lp->orig_lowbo[i] != 0)
                mpq_div(lp->orig_lowbo[i], lp->orig_lowbo[i], scalechange[i]);//lp->orig_lowbo[i] /= scalechange[i];
            if(mpq_cmp(lp->orig_upbo[i], lp->infinite) != 0)//lp->orig_upbo[i] != lp->infinite)
                mpq_div(lp->orig_upbo[i], lp->orig_upbo[i], scalechange[i]);//lp->orig_upbo[i] /= scalechange[i];

            /*VS - Since we don't use scalechange[i] anymore, clear the mpq */
            mpq_clear(scalechange[i]);
        }
        lp->columns_scaled=TRUE;
    }
    free(scalechange);
    lp->scaling_used = TRUE;
    lp->eta_valid = FALSE;
}

void reset_basis(lprec *lp)
{
    lp->basis_valid=FALSE;
}

void write_solution(lprec *lp, FILE *stream )
/* Added 7/23/98 by Anthony R. Cassandra
   
   There was no 'write_solution()' routine, so I adapted the
   'print_solution()' routine to take a FILE as input, renamed it
   write_solution() and added a print_solution() that just calls this
   with stdout.
*/
{
    int i;

    /*fprintf(stream, "Value of objective function: %16.10g\n",
            (double)lp->best_solution[0]);*/
    fprintf(stream, "Value of objective function: ");
    mpq_out_str(stream, 10, lp->best_solution[0]);
    fprintf(stream, "\n");

    /* print normal variables */
    for(i = 1; i <= lp->columns; i++) {
        if (lp->names_used)
            /*fprintf(stream, "%-10s%16.5g\n", lp->col_name[i],
                    (double)lp->best_solution[lp->rows+i]);*/
            fprintf(stream, "%-10s", lp->col_name[i]);
        else
            /*fprintf(stream, "Var [%4d]  %16.5g\n", i,
                    (double)lp->best_solution[lp->rows+i]);*/
            fprintf(stream, "Var [%4d]  ", i);
        mpq_out_str(stream, 10, lp->best_solution[lp->rows+i]);
        fprintf(stream, "\n");
    }

    /* print achieved constraint values */
    if(lp->verbose)
    {
        fprintf(stream, "\nActual values of the constraints:\n");
        for(i = 1; i <= lp->rows; i++) {
            if (lp->names_used)
                /*fprintf(stream, "%-10s%16.5g\n", lp->row_name[i],
                        (double)lp->best_solution[i]);*/
                fprintf(stream, "%-10s", lp->row_name[i]);
            else
                /*fprintf(stream, "Row [%4d]  %16.5g\n", i,
                        (double) lp->best_solution[i]);*/
                fprintf(stream, "Row [%4d]  ", i);
            mpq_out_str(stream, 10, lp->best_solution[i]);
            fprintf(stream, "\n");
        }
    }

    if((lp->verbose || lp->print_duals))
    {
        if(lp->max_level != 1)
            fprintf(stream,
                    "These are the duals from the node that gave the optimal solution.\n");
        else
            fprintf(stream, "\nDual values:\n");
        for(i = 1; i <= lp->rows; i++) {
            if (lp->names_used)
                /*fprintf(stream, "%-10s%16.5g\n", lp->row_name[i],
                        (double)lp->duals[i]);*/
                fprintf(stream, "%-10s", lp->row_name[i]);
            else
                /*fprintf(stream, "Row [%4d]  %16.5g\n", i, (double)lp->duals[i]);*/
                fprintf(stream, "Row [%4d]  ", i);
            mpq_out_str(stream, 10, lp->duals[i]);
            fprintf(stream, "\n");
        }
    }
    fflush(stream);
} /* write_solution */

void print_solution(lprec *lp)
/* Modifed 7/23/98 by Anthony R. Cassandra
   All the functionality that was here is moved to the
   write_solution() routine and this now just calls that with stdout
   as an input parameter.
*/
{
    write_solution( lp, stdout );
} /* PrintSolution */

void write_LP(lprec *lp, FILE *output)
{
    int i, j;
    REAL *row;
    REAL neg_one;
    REAL pos_one; //neg_one and pos_one only used for comparisons
    REAL temp;
    mpq_init(neg_one);
    mpq_set_si(neg_one, -1, 1);
    mpq_init(pos_one);
    mpq_set_ui(pos_one, 1, 1);
    mpq_init(temp);
    MALLOC(row, lp->columns+1);
    if(lp->maximise)
        fprintf(output, "max:");
    else
        fprintf(output, "min:");

    get_row(lp, 0, row);
    for(i = 1; i <= lp->columns; i++)
        if(mpq_sgn(row[i]) != 0)
        {
            if(mpq_cmp(row[i], neg_one) == 0)//row[i] == -1)
                fprintf(output, " -");
            else if(mpq_cmp(row[i], pos_one))//row[i] == 1)
                fprintf(output, " +");
            else {
                fprintf(output, " ");
                mpq_out_str(output, 10, row[i]);//fprintf(output, " %+g ", row[i]);
            }
            if(lp->names_used)
                fprintf(output, "%s", lp->col_name[i]);
            else
                fprintf(output, "x%d", i);
        }
    fprintf(output, ";\n");

    for(j = 1; j <= lp->rows; j++)
    {
        if(lp->names_used)
            fprintf(output, "%s:", lp->row_name[j]);
        get_row(lp, j, row);
        for(i = 1; i <= lp->columns; i++) {
            if (mpq_sgn(row[i]) != 0) {
                if (mpq_cmp(row[i], neg_one) == 0)//row[i] == -1)
                    fprintf(output, " -");
                else if (mpq_cmp(row[i], pos_one))//row[i] == 1)
                    fprintf(output, " +");
                else {
                    fprintf(output, " ");
                    mpq_out_str(output, 10, row[i]);//fprintf(output, " %+g ", row[i]);
                }
                if (lp->names_used)
                    fprintf(output, "%s", lp->col_name[i]);
                else
                    fprintf(output, "x%d", i);
            }
            /* VS - don't need row[i] anymore, so deallocate it. */
            mpq_clear(row[i]);
        }
        if(mpq_sgn(lp->orig_upbo[j]) == 0)
            fprintf(output, " =");
        else if(lp->ch_sign[j])
            fprintf(output, " >");
        else
            fprintf(output, " <");
        if(lp->ch_sign[j]) {
            //fprintf(output, " %g;\n", -lp->orig_rh[j]);
            fprintf(output, " ");
            mpq_neg(temp, lp->orig_rh[j]);
            mpq_out_str(output, 10, temp);
            fprintf(output, ";\n");
        }
        else {
            //fprintf(output, " %g;\n", lp->orig_rh[j]);
            fprintf(output, " ");
            mpq_out_str(output, 10, lp->orig_rh[j]);
            fprintf(output, ";\n");
        }
    }
    for(i = lp->rows+1; i <= lp->sum; i++)
    {
        if(mpq_sgn(*lp->orig_lowbo[i]) != 0)
        {
            if(lp->names_used) {
                //fprintf(output, "%s > %g;\n", lp->col_name[i - lp->rows],
                        //lp->orig_lowbo[i]);
                fprintf(output, "%s > ", lp->col_name[i - lp->rows]);
                mpq_out_str(output, 10, lp->orig_lowbo[i]);
                fprintf(output, ";\n");
            }
            else {
                //fprintf(output, "x%d > %g;\n", i - lp->rows,
                        //lp->orig_lowbo[i]);
                fprintf(output, "x%d > ", i - lp->rows);
                mpq_out_str(output, 10, lp->orig_lowbo[i]);
                fprintf(output, ";\n");
            }
        }
        if(mpq_cmp(*lp->orig_upbo[i], lp->infinite) != 0)
        {
            if(lp->names_used) {
                //fprintf(output, "%s < %g;\n", lp->col_name[i - lp->rows],
                        //lp->orig_upbo[i]);
                fprintf(output, "%s < ", lp->col_name[i - lp->rows]);
                mpq_out_str(output, 10, lp->orig_upbo[i]);
                fprintf(output, ";\n");
            }
            else {
                //fprintf(output, "x%d < %g;\n", i - lp->rows, lp->orig_upbo[i]);
                fprintf(output, "x%d < ", i - lp->rows);
                mpq_out_str(output, 10, lp->orig_upbo[i]);
            }
        }
    }


    i=1;
    while(!lp->must_be_int[lp->rows+i]  && i <= lp->columns)
        i++;
    if(i <= lp->columns)
    {
        if(lp->names_used)
            fprintf(output, "\nint %s", lp->col_name[i]);
        else
            fprintf(output, "\nint x%d", i);
        i++;
        for(; i <= lp->columns; i++)
            if(lp->must_be_int[lp->rows+i]) {
                if(lp->names_used)
                    fprintf(output, ",%s", lp->col_name[i]);
                else
                    fprintf(output, ", x%d", i);
            }
        fprintf(output, ";\n");
    }

    mpq_clear(temp);
    mpq_clear(neg_one);
    mpq_clear(pos_one);


    free(row);
}




void write_MPS(lprec *lp, FILE *output)
{
    int i, j, marker, putheader;
    REAL *column;
    REAL a;
    REAL temp;
    mpq_init(*a);
    mpq_init(*temp);


    MALLOC(column, lp->rows+1);
    marker=0;
    fprintf(output, "NAME          %s\n", lp->lp_name);
    fprintf(output, "ROWS\n");
    for(i = 0; i <= lp->rows; i++)
    {
        if(i==0)
            fprintf(output, " N  ");
        else
        if(lp->orig_upbo[i] != 0) {
            if(lp->ch_sign[i])
                fprintf(output, " G  ");
            else
                fprintf(output, " L  ");
        }
        else
            fprintf(output, " E  ");
        if(lp->names_used)
            fprintf(output, "%s\n", lp->row_name[i]);
        else
            fprintf(output, "r_%d\n", i);
    }

    fprintf(output, "COLUMNS\n");
    j = 0;
    for(i = 1; i <= lp->columns; i++)
    {
        if((lp->must_be_int[i+lp->rows]) && (marker % 2)==0)
        {
            fprintf(output,
                    "    MARK%04d  'MARKER'                 'INTORG'\n",
                    marker);
            marker++;
        }
        if((!lp->must_be_int[i+lp->rows]) && (marker % 2)==1)
        {
            fprintf(output,
                    "    MARK%04d  'MARKER'                 'INTEND'\n",
                    marker);
            marker++;
        }
        get_column(lp, i, column);
        j=0;
        if(lp->maximise) //TODO: Can probably clean up this big if/else. No reason for so much repeated code.
        {
            if(mpq_sgn(column[j]) != 0)//column[j] != 0)
            {
                mpq_neg(temp, column[j]); //since both cases below use -column[j]
                if(lp->names_used) {
                    //fprintf(output, "    %-8s  %-8s  %g\n", lp->col_name[i],
                    //lp->row_name[j], -column[j]);
                    fprintf(output, "    %-8s  %-8s  ", lp->col_name[i], lp->row_name[j]);
                    mpq_out_str(output, 10, temp);
                    fprintf(output, "\n");
                }
                else {
                    //fprintf(output, "    var_%-4d  r_%-6d  %g\n", i, j,
                            //-column[j]);
                    fprintf(output,"    var_%-4d  r_%-6d ", i, j);
                    mpq_out_str(output, 10, temp);
                    fprintf(output, "\n");
                }
            }
        }
        else
        {
            if(mpq_sgn(*column[j]) != 0)//column[j] != 0)
            {
                if(lp->names_used) {
                    //fprintf(output, "    %-8s  %-8s  %g\n", lp->col_name[i],
                            //lp->row_name[j], column[j]);
                    fprintf(output, "    %-8s  %-8s ", lp->col_name[i], lp->row_name[j]);
                    mpq_out_str(output, 10, column[j]);
                    fprintf(output, "\n");
                }
                else {
                    //fprintf(output, "    var_%-4d  r_%-6d  %g\n", i, j,
                        //column[j]);
                    fprintf(output, "    var_%-4d  r_%-6d ", i, j);
                    mpq_out_str(output, 10, column[j]);
                    fprintf(output, "\n");
                }

            }
        }
        for(j=1; j <= lp->rows; j++) {
            if (column[j] != 0) {
                if (lp->names_used) {
                    //fprintf(output, "    %-8s  %-8s  %g\n", lp->col_name[i],
                    //lp->row_name[j], column[j]);
                    fprintf(output, "    %-8s  %-8s ", lp->col_name[i], lp->row_name[j]);
                }
                else {
                    //fprintf(output, "    var_%-4d  r_%-6d  %g\n", i, j,
                    //column[j]);
                    fprintf(output, "    var_%-4d  r_%-6d ", i, j);
                }
                mpq_out_str(output, 10, column[j]);
                fprintf(output, "\n");
            }
            /* VS - clear column[j] here since we won't use it again.*/
            mpq_clear(column[j]);
        }
    }
    if((marker % 2) ==1)
    {
        fprintf(output, "    MARK%04d  'MARKER'                 'INTEND'\n",
                marker);
        marker++;
    }

    fprintf(output, "RHS\n");
    for(i = 1; i <= lp->rows; i++)
    {
        mpq_set(a, lp->orig_rh[i]);//a = lp->orig_rh[i];
        if(lp->scaling_used)
            mpq_div(a, a, lp->scale[i]);//a /= lp->scale[i];

        if(lp->ch_sign[i])
        {
            mpq_neg(temp, a);
            if(lp->names_used)
                //fprintf(output, "    RHS       %-8s  %g\n", lp->row_name[i],
                        //(double)-a);
                fprintf(output, "    RHS       %-8s  ", lp->row_name[i]);
            else
                //fprintf(output, "    RHS       r_%-6d  %g\n", i,
                        //(double)-a);
                fprintf(output, "    RHS       r_%-6d  ", i);
            mpq_out_str(output, 10, temp);
            fprintf(output, "\n");
        }
        else
        {
            if(lp->names_used)
                //fprintf(output, "    RHS       %-8s  %g\n", lp->row_name[i],
                        //(double)a);
                fprintf(output, "    RHS       %-8s  ", lp->row_name[i]);
            else
                //fprintf(output, "    RHS       r_%-6d  %g\n", i,
                        //(double)a);
                fprintf(output, "    RHS       r_%-6d  ", i);
            mpq_out_str(output, 10, a);
            fprintf(output, "\n");
        }
    }

    putheader = TRUE;
    for(i = 1; i <= lp->rows; i++)
        if(mpq_cmp(lp->orig_upbo[i], lp->infinite) != 0 && mpq_sgn(lp->orig_upbo[i]) != 0) {//(lp->orig_upbo[i] != lp->infinite) && (lp->orig_upbo[i] != 0.0)) {
            if(putheader)
            {
                fprintf(output, "RANGES\n");
                putheader = FALSE;
            }
            mpq_set(a, lp->orig_upbo[i]);//a = lp->orig_upbo[i];
            if(lp->scaling_used)
                mpq_div(a, a, lp->scale[i]);//a /= lp->scale[i];
            if(lp->names_used)
                /*fprintf(output, "    RGS       %-8s  %g\n", lp->row_name[i],
                        (double)a);*/
                fprintf(output, "    RGS       %-8s  ", lp->row_name[i]);
            else
                /*fprintf(output, "    RGS       r_%-6d  %g\n", i,
                        (double)a);*/
                fprintf(output, "    RGS       r_%-6d  ", i);
            mpq_out_str(output, 10, a);
            fprintf(output, "\n");
        }
        else if(mpq_sgn(lp->orig_lowbo[i]) != 0){//(lp->orig_lowbo[i] != 0.0)) {
            if(putheader)
            {
                fprintf(output, "RANGES\n");
                putheader = FALSE;
            }
            mpq_set(a, lp->orig_lowbo[i]);//a = lp->orig_lowbo[i];
            if(lp->scaling_used)
                mpq_div(a, a, lp->scale[i]);//a /= lp->scale[i];

            mpq_neg(temp, a); //since both cases below use -a
            if(lp->names_used)
                //fprintf(output, "    RGS       %-8s  %g\n", lp->row_name[i],
                        //(double)-a);
                fprintf(output, "    RGS       %-8s  ", lp->row_name[i]);
            else
                //fprintf(output, "    RGS       r_%-6d  %g\n", i,
                        //(double)-a);
                fprintf(output, "    RGS       r_%-6d  ", i);
            mpq_out_str(output, 10, temp);
            fprintf(output, "\n");
        }

    fprintf(output, "BOUNDS\n");
    if(lp->names_used)
        for(i = lp->rows + 1; i <= lp->sum; i++)
        {
            if(mpq_sgn(lp->orig_lowbo[i]) != 0 && mpq_cmp(lp->orig_upbo[i], lp->infinite) < 0 &&//(lp->orig_lowbo[i] != 0) && (lp->orig_upbo[i] < lp->infinite) &&
                (mpq_equal(lp->orig_lowbo[i], lp->orig_upbo[i])))//(lp->orig_lowbo[i] == lp->orig_upbo[i]))
            {
                mpq_set(*a, *lp->orig_upbo[i]);//a = lp->orig_upbo[i];
                if(lp->scaling_used)
                    mpq_mul(*a, *a, *lp->scale[i]);//a *= lp->scale[i];

                //fprintf(output, " FX BND       %-8s  %g\n",
                        //lp->col_name[i- lp->rows], (double)a);
                fprintf(output, " FX BND       %-8s  ", lp->col_name[i - lp->rows]);
                mpq_out_str(output, 10, *a);
                fprintf(output, "\n");
            }
            else
            {
                if(mpq_cmp(*lp->orig_upbo[i], *lp->infinite) < 0){//lp->orig_upbo[i] < lp->infinite) {
                    mpq_set(*a, *lp->orig_upbo[i]);//a = lp->orig_upbo[i];

                    if(lp->scaling_used)
                        mpq_mul(*a, *a, *lp->scale[i]);//a *= lp->scale[i];

                    //fprintf(output, " UP BND       %-8s  %g\n",
                            //lp->col_name[i- lp->rows], (double)a);
                    fprintf(output, " UP BND       %-8s  ", lp->col_name[i - lp->rows]);
                    mpq_out_str(output, 10, *a);
                }
                if(mpq_sgn(*lp->orig_lowbo[i]) != 0) {//lp->orig_lowbo[i] != 0) {
                    mpq_set(*a, *lp->orig_lowbo[i]);//a = lp->orig_lowbo[i];
                    if(lp->scaling_used)
                        mpq_mul(*a, *a, *lp->scale[i]);//a *= lp->scale[i];

                    //fprintf(output, " LO BND       %-8s  %g\n",
                            //lp->col_name[i- lp->rows], (double)lp->orig_lowbo[i]);
                    fprintf(output, " LO BND       %-8s  ", lp->col_name[i - lp->rows]);
                    mpq_out_str(output, 10, *lp->orig_lowbo[i]);
                }
            }
        }
    else
        for(i = lp->rows + 1; i <= lp->sum; i++)
        {
            if(mpq_sgn(*lp->orig_lowbo[i]) != 0 && mpq_cmp(*lp->orig_upbo[i], *lp->infinite) < 0 &&//(lp->orig_lowbo[i] != 0) && (lp->orig_upbo[i] < lp->infinite) &&
                    mpq_equal(*lp->orig_lowbo[i], *lp->orig_upbo[i]))//(lp->orig_lowbo[i] == lp->orig_upbo[i]))
            {
                mpq_set(*a, *lp->orig_upbo[i]);//a = lp->orig_upbo[i];
                if(lp->scaling_used)
                    mpq_mul(*a, *a, *lp->scale[i]);//a *= lp->scale[i];
                //fprintf(output, " FX BND       %-8s  %g\n",
                        //lp->col_name[i- lp->rows], (double)a);
                fprintf(output, " FX BND       %-8s  ", lp->col_name[i - lp->rows]);
                mpq_out_str(output, 10, *a);
                fprintf(output, "\n");
            }
            else
            {
                if(mpq_cmp(*lp->orig_upbo[i], *lp->infinite) < 0){//lp->orig_upbo[i] < lp->infinite) {
                    mpq_set(*a, *lp->orig_upbo[i]);//a = lp->orig_upbo[i];
                    if(lp->scaling_used)
                        mpq_mul(*a, *a, *lp->scale[i]);//a *= lp->scale[i];

                    //fprintf(output, " UP BND       var_%-4d  %g\n",
                            //i - lp->rows, (double)a);
                    fprintf(output, " UP BND       var_%-4d  ", i - lp->rows);
                    mpq_out_str(output, 10, *a);
                    fprintf(output, "\n");
                }
                if(mpq_sgn(*lp->orig_lowbo[i]) != 0){//lp->orig_lowbo[i] != 0) {
                    mpq_set(*a, *lp->orig_lowbo[i]);//a = lp->orig_lowbo[i];
                    if(lp->scaling_used)
                        mpq_mul(*a, *a, *lp->scale[i]);//a *= lp->scale[i];

                    //fprintf(output, " LO BND       var_%-4d  %g\n", i - lp->rows,
                            //(double)a);
                    fprintf(output, " LO BND       var_%-4d  ", i - lp->rows);
                    mpq_out_str(output, 10, *a);
                    fprintf(output, "\n");
                }
            }
        }
    fprintf(output, "ENDATA\n");
    mpq_clear(*a);
    mpq_clear(*temp);
    //column already cleared, can just free it.
    free(column);
}

void print_duals(lprec *lp)
{
    int i;
    for(i = 1; i <= lp->rows; i++) {
        if (lp->names_used)
            //fprintf(stdout, "%10s [%3d] % 10.4f\n", lp->row_name[i], i,
               //lp->duals[i]);
            fprintf(stdout, "%10s [%3d] ", lp->row_name[i], i);
        else
            //fprintf(stdout, "Dual       [%3d] % 10.4f\n", i, lp->duals[i]);
            fprintf(stdout, "Dual       [%3d] ", i);
        mpq_out_str(stdout, 10, *lp->duals[i]);
        fprintf(stdout, "\n");
    }
}

void print_scales(lprec *lp)
{
    int i;
    if(lp->scaling_used)
    {
        for(i = 0; i <= lp->rows; i++) {
            //fprintf(stdout, "Row[%3d]    scaled at % 10.6f\n", i, lp->scale[i]);
            fprintf(stdout, "Row[%3d]    scaled at ", i);
            mpq_out_str(stdout, 10, *lp->scale[i]);
        }
        for(i = 1; i <= lp->columns; i++) {
            //fprintf(stdout, "Column[%3d] scaled at % 10.6f\n", i,
                    //lp->scale[lp->rows + i]);
            fprintf(stdout, "Column[%3d] scaled at ", i);
            mpq_out_str(stdout, 10, *lp->scale[lp->rows + i]);
        }
    }
}

REAL my_mpq_min(REAL x, REAL y)
{
    if(mpq_cmp(*x, *y) < 0)
        return x;
    return y;
}

REAL my_mpq_max(REAL x, REAL y)
{
    if(mpq_cmp(*x, *y) > 0)
        return x;
    return y;
}
