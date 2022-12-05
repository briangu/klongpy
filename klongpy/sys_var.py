import sys

from .core import KGChannel, KGChannelDir


def eval_sys_var_arguments():
    """

        .a                                                   [Arguments]

        This variable holds a list of strings containing the command
        line arguments passed to a Klong program.

    """
    return sys.argv


def eval_sys_var_cin():
    """

        .cin                                             [Input-Channel]

        These variables are bound to the standard input (.cin), standard
        output (.cout), and standard error (.cerr) channels of the Klong
        process. They can be selected for input or output using the
        From-Channel (.fc) and To-Channel (.tc) functions.

        The standard I/O channels cannot be closed.

    """
    return KGChannel(sys.stdin, KGChannelDir.INPUT)


def eval_sys_var_cout():
    """

        .cout                                           [Output-Channel]

        See [Input-Channel]

    """
    return KGChannel(sys.stdout, KGChannelDir.OUTPUT)



def eval_sys_var_cerr():
    """

        .cerr                                            [Error-Channel]

        See [Input-Channel]

    """
    return KGChannel(sys.stderr, KGChannelDir.OUTPUT)


def eval_sys_var_cols():
    """

        .cols                                                  [Columns]

        This variable stores the number of columns of the screen on
        which the Klong session is running. It defaults to 80. This
        variable is used by the line editor, if compiled in and enabled
        (see .edit).

    """
    return 80


def eval_sys_var_epsilon():
    """

        .e                                                     [Epsilon]

        .e is the smallest value by which two real numbers (0.1<=x<1)
        can differ. For numbers smaller than 0.1, there would be a
        smaller difference and number x>=1 cannot differ by .e, because
        1+.e --> 1. the actual value of .e is implementation-dependent.
        On a 9-digit implementation, it would be 0.000000001 (1e-9).

        The logarithm to base 10 of %.e (ln(%.e)%ln(10)) equals the
        number of digits in the mantissa of a real number (this is
        exactly the exponent in the scientific notation of %.e).

    """
    return 0.000000000000000001


def eval_sys_var_edit():
    """

        .edit                                                   [Editor]

        When this variable set to a true value AND the line editor is
        compiled into the Klong executable, then line editing and
        history will be enabled on the Klong prompt. See the section on
        LINE EDITING for details.

    """
    pass


# def eval_sys_var_f():
#     """

#         .f                                                    [Function]

#         The variable .f is always bound to the function that is currently
#         being computed, so it allows you to write anonymous recursive
#         functions:

#         {:[0=x;[];1,.f(x-1)]}

#         Note that .f is lexically bound to the innermost function, so

#         {:[@x;0;1+|/{.f(x)}'x]}
#                      ^^^^^
#         would diverge. (But the effect here is due to unnecessary eta
#         expansion; {:[@x;0;1+|/.f'x]} would work fine.)

#     """
#     pass


def eval_sys_var_fastpow():
    """

        .fastpow                                            [Fast-Power]

        When this variable set to a true value, then expressions of the
        form x^y will compile to .pow(x;y), which makes computations
        involving powers of real numbers much faster (about eight times
        on the author's hardware).

        Setting .fastpow::0 will disable this feature and restore the
        previous behavior of Klong, just in case.

    """
    pass


def eval_sys_var_host():
    """

        .h                                                        [Host]

        This variable holds a unique symbol identifying the host system
        running the Klong process. Currently, this is either :unix or
        :plan9.

    """
    pass


# def eval_sys_var_it():
#     """

#         it                                                          [It]

#         In interactive mode, "it" always holds the value of the most
#         recent successful computation. See also: INTERACTION, below.

#     """
#     pass

