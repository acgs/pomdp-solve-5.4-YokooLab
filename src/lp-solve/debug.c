#include "lpkit.h"
#include "lpglob.h"
#include <stdarg.h>
#include <gmp.h>

static void print_indent(void)
{
  int i;

  fprintf(stderr, "%2d", Level);
  if(Level < 50) /* useless otherwise */
    for(i = Level; i > 0; i--)
      fprintf(stderr, "--");
  else
    fprintf(stderr, " *** too deep ***");
  fprintf(stderr, "> ");
} /* print_indent */


void debug_print_solution(lprec *lp)
{
  int i;

  if(lp->debug)
      for (i = lp->rows + 1; i <= lp->sum; i++)
      {
          print_indent();
          if (lp->names_used) {
              fprintf(stderr, "%-10s", lp->col_name[i - lp->rows]);
              mpq_out_str(stderr, 10, lp->solution[i]);
              fprintf(stderr, "\n");
          }
          else{
              fprintf(stderr, "Var[%5d]   ", i - lp->rows);
              mpq_out_str(stderr, 10, lp->solution[i]);
              fprintf(stderr, "\n");
          }
      }
} /* debug_print_solution */


void debug_print_bounds(lprec *lp, REAL *upbo, REAL *lowbo)
{
  int i;

  if(lp->debug)
      for(i = lp->rows + 1; i <= lp->sum; i++)
      {
          if(mpq_equal(lowbo[i], upbo[i]))//lowbo[i] == upbo[i])
          {
              print_indent();
              if (lp->names_used) {
                  fprintf(stderr, "%s = ", lp->col_name[i - lp->rows]);
                  mpq_out_str(stderr, 10, lowbo[i]);
                  fprintf(stderr, "\n");
              }
            else {
                  fprintf(stderr, "Var[%5d]  = ", i - lp->rows);
                  mpq_out_str(stderr, 10, lowbo[i]);
                  fprintf(stderr, "\n");
            }

	  }
	else
	  {
	    if(mpq_sgn(lowbo[i]) != 0)//lowbo[i] != 0) /* VS - change comparison to use mpq */
        {
		    print_indent();
		    if (lp->names_used) {
                fprintf(stderr, "%s > ", lp->col_name[i - lp->rows]);
                mpq_out_str(stderr, 10, lowbo[i]);
                fprintf(stderr, "\n");
            }
		    else {
                fprintf(stderr, "Var[%5d]  > ", i - lp->rows);
                mpq_out_str(stderr, 10, lowbo[i]);
                fprintf(stderr, "\n");
            }
	      }
	    if(upbo[i] != lp->infinite)
        {
		    print_indent();
		    if (lp->names_used) {
                fprintf(stderr, "%s < ", lp->col_name[i - lp->rows]);
                mpq_out_str(stderr, 10, upbo[i]);
                fprintf(stderr, "\n");
            }
		    else {
                fprintf(stderr, "Var[%5d]  < ", i - lp->rows);
                mpq_out_str(stderr, 10, upbo[i]);
                fprintf(stderr, "\n");
            }
        }
	  }
      }
} /* debug_print_bounds */


void debug_print(lprec *lp, char *format, ...)
{
  va_list ap;

  if(lp->debug)
    {
      va_start(ap, format);
      print_indent();
      vfprintf(stderr, format, ap);
      fputc('\n', stderr);
      va_end(ap);
    }
} /* debug_print */

