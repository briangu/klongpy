/*
 * Klong interpreter
 * Nils M Holm, 2015--2019
 * In the public domain
 *
 * Under jurisdictions without a public domain, the CC0 applies:
 * https://creativecommons.org/publicdomain/zero/1.0/
 */

#include "s9core.h"
#include "s9import.h"

#define VERSION		"20190926"

#ifdef plan9
 #define handle_sigquit()
 #define handle_sigint()	notify(keyboard_interrupt)
#else
 #include <time.h>
 #include <signal.h>
 #define handle_sigquit()	signal(SIGQUIT, keyboard_quit)
 #define handle_sigint()	signal(SIGINT, keyboard_interrupt)
#endif

#define TOKEN_LENGTH	256
#define MIN_DICT_LEN	13

#define NTRACE		10

#define DFLPATH		".:lib"

#define T_DICTIONARY	(USER_SPECIALS-1)
#define T_VARIABLE	(USER_SPECIALS-2)
#define T_PRIMOP	(USER_SPECIALS-3)
#define T_BARRIER	(USER_SPECIALS-4)
#define STRING_NIL	(USER_SPECIALS-5)
#define NO_VALUE	(USER_SPECIALS-6)

#define list_p(x) \
	(pair_p(x) || NIL == (x))

#define dictionary_p(n) \
        (!special_p(n) && (tag(n) & S9_ATOM_TAG) && car(n) == T_DICTIONARY)
#define dict_data(x)	cddr(x)
#define dict_table(x)	vector(dict_data(x))
#define dict_len(x)	vector_len(cddr(x))
#define dict_size(x)	cadr(x)

#define fun_immed(x)	cadr(x)
#define fun_arity(x)	caddr(x)
#define fun_body(x)	cdddr(x)

#define variable_p(n) \
        (!special_p(n) && (tag(n) & S9_ATOM_TAG) && car(n) == T_VARIABLE)
#define var_symbol(x)	cadr(x)
#define var_name(x)	symbol_name(var_symbol(x))
#define var_value(x)	cddr(x)

#define primop_p(n) \
        (!special_p(n) && (tag(n) & S9_ATOM_TAG) && car(n) == T_PRIMOP)
#define primop_slot(x)	cadr(x)

#define syntax_body(x)	cdr(x)

cell	Dstack;
cell	Sys_dict;
cell	Safe_dict;
cell	Frame;
int	State;
cell	Tmp;
cell	Locals;
cell	Barrier;
cell	S, F;
cell	Prog, P;
cell	Tok, T;
cell	Epsilon_var;
cell	Loading;
cell	Module;
cell	Mod_funvars;
char	Modname[TOKEN_LENGTH+1];
cell	Locnames;
cell	Local_id;
int	To_chan, From_chan;
int	Prog_chan;
cell	Trace[NTRACE];
int	Traceptr;
int	Report;
int	Quiet;
int	Script;
int	Debug;
int	Transcript;
char	Inbuf[TOKEN_LENGTH+1];
int	Listlev;
int	Incond;
int	Infun;
int	Line;
int	Display;
char	Image_path[TOKEN_LENGTH+13];

volatile int	Intr;

/* VM opcode symbols */

cell	S_amend, S_amendd, S_argv, S_atom, S_call0, S_call1, S_call2,
	S_call3, S_char, S_clear, S_conv, S_cut, S_def, S_div, S_down,
	S_drop, S_each, S_each2, S_eachl, S_eachp, S_eachr, S_enum,
	S_eq, S_expand, S_find, S_first, S_floor, S_form, S_format,
	S_format2, S_fun0, S_fun1, S_fun2, S_fun3, S_group, S_gt,
	S_host, S_if, S_imm1, S_imm2, S_index, S_indexd, S_intdiv,
	S_it, S_iter, S_join, S_list, S_lslit, S_lt, S_match, S_max,
	S_min, S_minus, S_newdict, S_neg, S_not, S_over, S_over2, S_plus,
	S_pop0, S_pop1, S_pop2, S_pop3, S_power, S_prog, S_range,
	S_recip, S_rem, S_reshape, S_rev, S_rot, S_sconv, S_siter,
	S_sover, S_sover2, S_swhile, S_shape, S_size, S_split, S_swap,
	S_syscall, S_take, S_thisfn, S_times, S_transp, S_up, S_undef,
	S_while, S_x, S_y, S_z;

enum Adverb_states {
	S_EVAL, S_APPIF, S_APPLY, S_EACH, S_EACH2, S_EACHL, S_EACHP,
	S_EACHR, S_OVER, S_CONV, S_ITER, S_WPRED, S_WEXPR, S_S_OVER,
	S_S_CONV, S_S_ITER, S_S_WPRED, S_S_WEXPR
};

struct OP_ {
	char	*name;
	int	syntax;
	void	(*handler)(void);
};

#define OP	struct OP_

struct SYS_ {
	char	*name;
	int	arity;
	void	(*handler)(void);
};

#define SYS	struct SYS_

cell *GC_root[] = {
	&S, &F, &Dstack, &Frame, &Sys_dict, &Safe_dict, &Tmp, &Barrier,
	&Tok, &Prog, &P, &T, &S_it, &Locals, &Module, &Mod_funvars,
	&Locnames, &Loading, NULL
};

cell *Image_vars[] = {
	&Barrier, &Dstack, &Epsilon_var, &F, &Frame, &Loading,
	&Local_id, &Locals, &Locnames, &Mod_funvars, &Module,
	&P, &Prog, &S, &Safe_dict, &Sys_dict, &T, &Tmp, &Tok,
	&S_amend, &S_amendd, &S_argv, &S_atom, &S_call0, &S_call1,
	&S_call2, &S_call3, &S_char, &S_clear, &S_conv, &S_cut, &S_def,
	&S_div, &S_down, &S_drop, &S_each, &S_each2, &S_eachl, &S_eachp,
	&S_eachr, &S_enum, &S_eq, &S_expand, &S_find, &S_first, &S_floor,
	&S_form, &S_format, &S_format2, &S_fun0, &S_fun1, &S_fun2, &S_fun3,
	&S_group, &S_gt, &S_host, &S_if, &S_imm1, &S_imm2, &S_index,
	&S_indexd, &S_intdiv, &S_it, &S_iter, &S_join, &S_list, &S_lslit,
	&S_lt, &S_match, &S_max, &S_min, &S_minus, &S_neg, &S_newdict,
	&S_not, &S_over, &S_over2, &S_plus, &S_pop0, &S_pop1, &S_pop2,
	&S_pop3, &S_power, &S_prog, &S_range, &S_recip, &S_rem, &S_reshape,
	&S_rev, &S_rot, &S_sconv, &S_siter, &S_sover, &S_sover2, &S_swhile,
	&S_shape, &S_size, &S_split, &S_swap, &S_syscall, &S_take,
	&S_thisfn, &S_times, &S_transp, &S_up, &S_undef, &S_while, &S_x,
	&S_y, &S_z,
	NULL
};

/*
 * Allocators
 */

/* From http://planetmath.org/goodhashtableprimes */

static int hashsize(int n) {
        if (n < 5) return 5;
	if (n < 11) return 11;
	if (n < 23) return 23;
	if (n < 47) return 47;
	if (n < 97) return 97;
	if (n < 193) return 193;
	if (n < 389) return 389;
	if (n < 769) return 769;
	if (n < 1543) return 1543;
	if (n < 3079) return 3079;
	if (n < 6151) return 6151;
	if (n < 12289) return 12289;
	if (n < 24593) return 24593;
	if (n < 49157) return 49157;
	if (n < 98317) return 98317;
	if (n < 196613) return 196613;
	if (n < 786433) return 786433;
	if (n < 1572869) return 1572869;
	if (n < 3145739) return 3145739;
	if (n < 6291469) return 6291469;
	if (n < 12582917) return 12582917;
	if (n < 25165843) return 25165843;
	if (n < 50331653) return 50331653;
	if (n < 100663319) return 100663319;
	if (n < 201326611) return 201326611;
	if (n < 402653189) return 402653189;
	if (n < 805306457) return 805306457;
	return 1610612741;
}

static cell make_dict(int k) {
	cell	d;
	int	i;

	k = hashsize(k);
	d = make_vector(k);
	for (i = 0; i < k; i++)
		vector(d)[i] = NIL;
	d = new_atom(0, d);
	d = new_atom(T_DICTIONARY, d);
	return d;
}

static cell make_function(cell body, int immed, int arity) {
	return new_atom(T_FUNCTION,
			new_atom(immed,
				new_atom(arity, body)));
}

static cell find_var(char *s) {
	cell	p;

	for (p = Sys_dict; p != NIL; p = cdr(p)) {
		if (strcmp(s, var_name(car(p))) == 0)
			return car(p);
	}
	return UNDEFINED;
}

static cell make_variable(char *s, cell v) {
	cell	n;
	char	name[TOKEN_LENGTH+1];

	n = find_var(s);
	if (n != UNDEFINED) return n;
	strcpy(name, s);
	save(v);
	n = symbol_ref(name);
	n = cons(n, v);
	n = new_atom(T_VARIABLE, n);
	Sys_dict = cons(n, Sys_dict);
	unsave(1);
	return n;
}

static cell var_ref(char *s) {
	return make_variable(s, NIL);
}

static cell make_primop(int slot, int syntax) {
	cell	n;

	n = new_atom(slot, NIL);
	return new_atom(syntax? T_SYNTAX: T_PRIMOP, n);
}

/*
 * Error handling
 */

void kg_write(cell x);

static void printtrace(void) {
	int	i, j;

	prints("kg: trace:");
	i = Traceptr;
	for (j=0; j<NTRACE; j++) {
		if (i >= NTRACE)
			i = 0;
		if (Trace[i] != UNDEFINED) {
			prints(" ");
			prints(symbol_name(Trace[i]));
		}
		i++;
	}
	nl();
}

static cell error(char *msg, cell arg) {
	int	p = set_output_port(Error_port);
	char	buf[100];

	Incond = 0;
	Infun = 0;
	Listlev = 0;
	if (0 == Report) {
		s9_abort();
		return UNDEFINED;
	}
	if (s9_aborted())
		return UNDEFINED;
	s9_abort();
	if (Quiet)
		set_output_port(Error_port);
	prints("kg: error: ");
	if (Loading != NIL) {
		kg_write(car(Loading));
		sprintf(buf, ": %d: ", Line);
		prints(buf);
	}
	prints(msg);
	if (arg != VOID) {
		prints(": ");
		kg_write(arg);
	}
	nl();
	if (Debug) {
		printtrace();
	}
	set_output_port(p);
	if (Quiet)
		bye(1);
	return UNDEFINED;
}

/*
 * Reader
 */

static char *kg_getline(char *s, int k) {
	int	c = 0;
	char	*p = s;

	while (k--) {
		c = readc();
		if (EOF == c || '\n' == c)
			break;
		*s++ = c;
	}
	Line++;
	*s = 0;
	return EOF == c && p == s? NULL: p;
}

#define is_white(c) \
	(' '  == (c) ||	\
	 '\t' == (c) ||	\
	 '\n' == (c) ||	\
	 '\r' == (c) ||	\
	 '\f' == (c))

#define is_symbolic(c) \
	(isalpha(c) || isdigit(c) || (c) == '.')

#define is_special(c) \
	(!isalpha(c) && !isdigit(c) && (c) >= ' ')

static int skip_white(void) {
	int	c;

	for (;;) {
		c = readc();
		while (is_white(c) && (NIL == Loading || c != '\n'))
			c = readc();
		if ('\n' == c)
			Line++;
		if (Loading != NIL && '\n' == c) {
			if (Listlev > 0 || Incond || Infun)
				continue;
			return EOF;
		}
		return c;
	}
}

static void comment(void) {
	int	c;

	c = readc();
	for (;;) {
		if (EOF == c) {
			error("missing end of comment", VOID);
			return;
		}
		if ('"' == c) {
			c = readc();
			if (c != '"') {
				rejectc(c);
				break;
			}
		}
		c = readc();
	}
}

static int skip(void) {
	int	c;

	for (;;) {
		c = skip_white();
		if (EOF == c) {
			if (input_port() < 0 && (Listlev || Incond || Infun))
			{
				if (!Quiet) {
					prints("        ");
					flush();
				}
				close_input_string();
				if (kg_getline(Inbuf, TOKEN_LENGTH) == NULL)
					return EOF;
				open_input_string(Inbuf);
			}
			else {
				return EOF;
			}
		}
		else if (':' == c) {
			c = readc();
			if (c != '"') {
				rejectc(c);
				return ':';
			}
			comment();
		}
		else {
			return c;
		}
	}
}

static cell read_string(void) {
	int	c, i;
	char	s[TOKEN_LENGTH+1];

	i = 0;
	c = readc();
	for (;;) {
		if (EOF == c)
			return error("missing end of string", VOID);
		if ('"' == c) {
			c = readc();
			if (c != '"') {
				rejectc(c);
				break;
			}
		}
		if (i < TOKEN_LENGTH)
			s[i++] = c;
		else if (TOKEN_LENGTH == i) {
			i++;
			error("string too long", make_string(s, i));
		}
		c = readc();
	}
	s[i] = 0;
	return make_string(s, i);
}

static void mkglobal(char *s) {
	char	*p;

	if ((p = strchr(s, '`')) != NULL)
		*p = 0;
}

static int is_local(char *s) {
	return strchr(s, '`') != NULL;
}

static int is_funvar(char *s) {
	cell	m, p;

	for (m = Mod_funvars; m != NIL; m = cdr(m)) {
		for (p = car(m); p != NIL; p = cdr(p)) {
			if (!strcmp(symbol_name(car(p)), s))
				return 1;
		}
	}
	return 0;
}

static int in_module(char *s) {
	cell	m;
	char	b[TOKEN_LENGTH+1];
	int	g;

	if (UNDEFINED == Module)
		return 0;
	if (is_funvar(s))
		return 0;
	g = !is_local(s);
	for (m = Module; m != NIL; m = cdr(m)) {
		strcpy(b, var_name(car(m)));
		if (g) {
			mkglobal(b);
		}
		if (!strcmp(s, b))
			return 1;
	}
	return 0;
}

static char *mklocal(char *s) {
	cell	loc, p;
	int	id;
	char	b[TOKEN_LENGTH+1];

	for (loc = Locnames; loc != NIL; loc = cdr(loc)) {
		id = caar(loc);
		for (p = cdar(loc); p != NIL; p = cdr(p)) {
			strcpy(b, symbol_name(car(p)));
			mkglobal(b);
			if (!strcmp(s, b)) {
				sprintf(s, "%s`%d", b, id);
				return s;
			}
		}
	}
	return NULL;
}

static void mkmodlocal(char *s) {
	if (strlen(s)+strlen(Modname) >= TOKEN_LENGTH-1)
		error("in-module symbol too long", make_string(s,strlen(s)));
	strcat(s, "`");
	strcat(s, Modname);
}

static cell read_sym(int c, int mod) {
	char	s[TOKEN_LENGTH+1];
	int	i;

	i = 0;
	while (is_symbolic(c)) {
		if (i < TOKEN_LENGTH)
			s[i++] = c;
		else if (TOKEN_LENGTH == i) {
			i++;
			error("symbol too long", make_string(s, i));
		}
		c = readc();
	}
	rejectc(c);
	s[i] = 0;
	if (mklocal(s) != NULL)
		;
	else if (0 == s[1] && ('x' == *s || 'y' == *s || 'z' == *s))
		;
	else if (mod &&
		 (in_module(s) ||
		 (Module != UNDEFINED && find_var(s) == UNDEFINED))
	) {
		mkmodlocal(s);
	}
	if (Listlev)
		return symbol_ref(s);
	return make_variable(s, NO_VALUE);
}

static cell read_num(int c) {
	char	s[TOKEN_LENGTH+1];
	int	i, c2 = 0;

	i = 0;
	if ('-' == c) {
		s[i++] = c;
		c = readc();
	}
	while (	isdigit(c) ||
		'.' == c ||
		'e' == c ||
		('e' == c2 && '+' == c) ||
		('e' == c2 && '-' == c))
	{
		if (i < TOKEN_LENGTH)
			s[i++] = c;
		else if (TOKEN_LENGTH == i) {
			i++;
			error("number too long", make_string(s, i));
		}
		c2 = c;
		c = readc();
	}
	rejectc(c);
	s[i] = 0;
	if (!string_numeric_p(s))
		return error("invalid number", make_string(s, i));
	return string_to_number(s);
}

static cell read_char(void) {
	return make_char(readc());
}

static cell read_xnum(int pre, int neg) {
	char	digits[] = "0123456789abcdef";
	char	buf[100];
	cell	base, num;
	int	c, p, nd;
	int	radix;

	c = pre;
	switch (c) {
	case 'b':
		radix = 2; break;
	case 'o':
		radix = 8; break;
	case 'x':
		radix = 16; break;
	default:
		radix = 10; break;
	}
	base = make_integer(radix);
	save(base);
	num = Zero;
	save(num);
	if (radix != 10)
		c = tolower(readc());
	nd = 0;
	while (1) {
		p = 0;
		while (digits[p] && digits[p] != c)
			p++;
		if (p >= radix) {
			if (0 == nd) {
				sprintf(buf, "invalid digit in #%c number",
					pre);
				unsave(2);
				return error(buf, make_char(c));
			}
			break;
		}
		num = bignum_multiply(num, base);
		car(Stack) = num;
		num = bignum_add(num, make_integer(p));
		car(Stack) = num;
		nd++;
		c = tolower(readc());
	}
	unsave(2);
	if (!nd) {
		sprintf(buf, "digits expected after #%c", pre);
		return error(buf, VOID);
	}
	rejectc(c);
	return neg? bignum_negate(num): num;
}

static cell kg_read(void);

static cell read_list(int dlm) {
	int	c, k = 0;
	cell	a, n;

	Listlev++;
	a = cons(NIL, NIL);
	save(a);
	c = skip();
	while (c != dlm) {
		rejectc(c);
		n = kg_read();
		if (eof_p(n)) {
			unsave(1);
			return error("unexpected end of list/dict", VOID);
		}
		car(a) = n;
		c = skip();
		if (c != dlm) {
			n = cons(NIL, NIL);
			cdr(a) = n;
			a = cdr(a);
		}
		k++;
	}
	Listlev--;
	a = unsave(1);
	if (0 == k)
		return NIL;
	return a;
}

static int string_hash(char *s) {
	unsigned	h = 0xdeadbeef;

	while (*s)
		h = ((h << 5) + h) ^ *s++;
	return abs((int) h);
}

static int hash(cell x, int k) {
	int	of;

	if (0 == k)
		return 0;
	if (integer_p(x))
		return abs(bignum_to_int(x, &of) % k);
	if (char_p(x))
		return char_value(x) % k;
	if (string_p(x))
		return abs(string_hash(string(x)) % k);
	if (symbol_p(x))
		return abs(string_hash(symbol_name(x)) % k);
	if (variable_p(x))
		return abs(string_hash(var_name(x)) % k);
	if (pair_p(x))
		return abs((length(x) * hash(car(x), k)) % k);
	return 0;
}

static int tuple_p(cell x) {
	return pair_p(x) && pair_p(cdr(x)) && NIL == cddr(x);
}

static cell list_to_dict(cell x) {
	cell	d, *v, e;
	int	n, k, i, h;

	save(x);
	n = length(x);
	k = hashsize(n);
	d = make_vector(k);
	save(d);
	v = vector(d);
	for (i = 0; i < k; i++)
		v[i] = NIL;
	while (x != NIL) {
		if (!tuple_p(car(x))) {
			unsave(2);
			return error("malformed dictionary entry", car(x));
		}
		h = hash(caar(x), k);
		v = vector(d);
		e = cons(car(x), v[h]);
		v = vector(d);
		v[h] = e;
		x = cdr(x);
	}
	unsave(2);
	d = new_atom(n, d);
	return new_atom(T_DICTIONARY, d);
}

static cell read_dict(void) {
	cell	x;

	x = read_list('}');
	return list_to_dict(x);
}

static cell shifted(void) {
	int	c;
	char	buf[3];
	cell	n;

	c = readc();
	if (isalpha(c) || '.' == c) {
		n = read_sym(c, 0);
		if (variable_p(n))
			return var_symbol(n);
		return n;
	}
	else if ('"' == c || isdigit(c)) {
		rejectc(c);
		return kg_read();
	}
	else if ('{' == c) {
		return read_dict();
	}
	else {
		buf[0] = ':';
		buf[1] = c;
		buf[2] = 0;
		return symbol_ref(buf);
	}
}

static cell read_op(int c) {
	char	buf[3];

	buf[1] = buf[2] = 0;
	buf[0] = c;
	if ('\\' == c) {
		c = readc();
		if ('~' == c || '*' == c)
			buf[1] = c;
		else
			rejectc(c);
	}
	return symbol_ref(buf);
}

#ifdef plan9
static int system(char *cmd) {
	int	r;
	Waitmsg	*w;

	r = fork();
	if (r < 0) {
		return -1;
	}
	else if (0 == r) {
		execl("/bin/rc", "/bin/rc", "-c", cmd, 0);
		exits("execl() failed");
	}
	else {
		w = wait();
		r = w->msg[0] != 0;
		free(w);
	}
	return r;
}
#endif

static void inventory(char *buf) {
	char	cmd[TOKEN_LENGTH+20], kpbuf[TOKEN_LENGTH+1];
	char	*p, *s;

#ifdef SAFE
	error("shell access disabled", VOID);
	return;
#endif
	if (buf[0]) {
		p = buf;
		sprintf(cmd, "cd %s; ls *.kg", buf);
	}
	else {
		p = getenv("KLONGPATH");
		if (NULL == p)
			p = DFLPATH;
		strncpy(kpbuf, p, TOKEN_LENGTH);
		kpbuf[TOKEN_LENGTH] = 0;
		p = kpbuf;
		if (strlen(p) >= TOKEN_LENGTH) {
			error("KLONGPATH too long!", VOID);
			return;
		}
		s = strchr(p, ':');
		if (s != NULL)
			*s = 0;
		sprintf(cmd, "cd %s; ls *.kg", p);
	}
	printf("%s:\n", p);
	system(cmd);
}

static void transcribe(cell x, int input) {
	cell	p;

	if (Transcript < 0)
		return;
	p = set_output_port(Transcript);
	if (input) {
		Display = 1;
		prints("\t");
	}
	kg_write(x);
	nl();
	Display = 0;
	set_output_port(p);
}

static void transcript(char *path) {
	if (Transcript >= 0) {
		close_port(Transcript);
		Transcript = -1;
		prints("transcript closed"); nl();
	}
	if (NULL == path || 0 == *path)
		return;
	if ((Transcript = open_output_port(path, 1)) < 0) {
		error("could not open transcript file",
			make_string(path, strlen(path)));
		return;
	}
	lock_port(Transcript);
	prints("sending transcript to: ");
	prints(path); nl();
}

static void eval(cell x);

static void apropos(char *s) {
	cell	x;

	if (0 == *s) {
		error("Usage: ']a function/operator' or ']a all'", VOID);
		return;
	}
	x = cons(S_pop1, NIL);
	x = cons(S_call1, x);
	x = cons(var_value(var_ref("help")), x);
	if (NIL == car(x)) {
		error("help function not loaded, try ]lhelp", VOID);
		return;
	}
	if (strcmp(s, "all") == 0)
		s = "";
	x = cons(make_string(s, strlen(s)), x);
	save(x);
	eval(x);
	unsave(1);
}

static cell load(cell x, int v, int scr);

static void meta_command(void) {
	int	cmd, c, i;
	char	buf[TOKEN_LENGTH];

	cmd = skip();
	c = skip();
	for (i=0; c != EOF; i++) {
		if (i < TOKEN_LENGTH-2)
			buf[i] = c;
		c = readc();
	}
	buf[i] = 0;
	switch (cmd) {
	case '!':
#ifdef SAFE
		error("shell access disabled", VOID);
#else
		system(buf);
#endif
		break;
	case 'a':
	case 'h':
		apropos(buf);
		break;
	case 'i':
		inventory(buf);
		break;
	case 'l':
		load(make_string(buf, strlen(buf)), 1, 0);
		open_input_string("");
		break;
	case 'q':
		prints("bye!"); nl();
		bye(0);
		break;
	case 't':
		transcript(buf);
		break;
	default:
		prints("! cmd     run shell command"); nl();
		prints("a fn/op   describe function/operator (apropos)"); nl();
		prints("i [dir]   inventory (of given directory)"); nl();
		prints("l file    load file.kg"); nl();
		prints("q         quit"); nl();
		prints("t [file]  transcript to file (none = off)"); nl();
		break;
	}
}

static cell kg_read(void) {
	int	c;

	c = skip();
	switch (c) {
	case '"':
		return read_string();
	case '0':
		c = tolower(readc());
		if ('c' == c)
			return read_char();
		if ('x' == c || 'o' == c || 'b' == c)
			return read_xnum(c, 0);
		rejectc(c);
		return read_num('0');
	case ':':
		return shifted();
	case '[':
		return read_list(']');
	case EOF:
		return END_OF_FILE;
	default:
		if ('-' == c && Listlev > 0) {
			c = readc();
			rejectc(c);
			if (isdigit(c))
				return read_num('-');
			return read_op('-');
		}
		if (	']' == c &&
			0 == Listlev &&
			0 == Incond &&
			0 == Infun &&
			NIL == Loading
		) {
			meta_command();
			return kg_read();
		}
		if (isdigit(c))
			return read_num(c);
		if (is_symbolic(c))
			return read_sym(c, 1);
		return read_op(c);
	}
}

/*
 * Printer
 */

static void write_char(cell x) {
	char	b[4];

	sprintf(b, Display? "%c": "0c%c", (int) char_value(x));
	prints(b);
}

static void write_string(cell x) {
	int	i, k;
	char	*s;
	char	b[2];

	if (Display) {
		blockwrite(string(x), string_len(x)-1);
		return;
	}
	b[1] = 0;
	prints("\"");
	k = string_len(x)-1;
	s = string(x);
	for (i = 0; i < k; i++) {
		if ('"' == s[i])
			prints("\"");
		b[0] = s[i];
		prints(b);
	}
	prints("\"");
}

static void write_list(cell x) {
	prints("[");
	while (x != NIL) {
		kg_write(car(x));
		if (cdr(x) != NIL)
			prints(" ");
		x = cdr(x);
	}
	prints("]");
}

static void write_dict(cell x) {
	int	k, i, first = 1;
	cell	*v, e;

	x = cddr(x);
	k = vector_len(x);
	prints(":{");
	v = vector(x);
	for (i = 0; i < k; i++) {
		if (NIL == v[i])
			continue;
		if (!first)
			prints(" ");
		first = 0;
		for (e = v[i]; e != NIL; e = cdr(e)) {
			kg_write(car(e));
			if (cdr(e) != NIL)
				prints(" ");
		}
	}
	prints("}");
}

static void write_chan(cell x) {
	char	b[100];

	sprintf(b, ":%schan.%d",
			input_port_p(x)? "in": "out",
			(int) port_no(x));
	prints(b);
}

static void write_primop(cell x) {
	char	b[100];

	sprintf(b, ":primop-%d", (int) primop_slot(x));
	prints(b);
}

static void write_fun(cell x) {
	if (0 == fun_arity(x))
		prints(":nilad");
	else if (1 == fun_arity(x))
		prints(":monad");
	else if (2 == fun_arity(x))
		prints(":dyad");
	else
		prints(":triad");
}

void kg_write(cell x) {
	if (NIL == x)
		prints("[]");
	else if (symbol_p(x)) {
		prints(":");
		prints(symbol_name(x));
	}
	else if (variable_p(x))
		prints(var_name(x));
	else if (undefined_p(x))
		prints(":undefined");
	else if (string_p(x))
		write_string(x);
	else if (integer_p(x))
		print_bignum(x);
	else if (char_p(x))
		write_char(x);
	else if (real_p(x))
		print_real(x);
	else if (eof_p(x))
		prints(":eof");
	else if (dictionary_p(x))
		write_dict(x);
	else if (input_port_p(x) || output_port_p(x))
		write_chan(x);
	else if (function_p(x))
		write_fun(x);
	else if (primop_p(x))
		write_primop(x);
	else if (x == Barrier)
		prints(":barrier");
	else if (x == STRING_NIL)
		prints(":stringnil");
	else
		write_list(x);
}

/*
 * Dictionaries
 */

static int dict_contains(cell x, cell y);

static int match(cell a, cell b) {
	int	k;

	if (a == b)
		return 1;
	if (real_p(a) && number_p(b)) {
		return real_approx_p(a, b);
	}
	if (number_p(a) && real_p(b)) {
		return real_approx_p(a, b);
	}
	if (number_p(a) && number_p(b)) {
		return real_equal_p(a, b);
	}
	if (symbol_p(a) && variable_p(b)) {
		return a == var_symbol(b);
	}
	if (variable_p(a) && symbol_p(b)) {
		return var_symbol(a) == b;
	}
	if (variable_p(a) && variable_p(b)) {
		return var_symbol(a) == var_symbol(b);
	}
	if (char_p(a) && char_p(b)) {
		return char_value(a) == char_value(b);
	}
	if (string_p(a) && string_p(b)) {
		k = string_len(a);
		if (string_len(b) == k)
			return memcmp(string(a), string(b), k) == 0;
		return 0;
	}
	if (pair_p(a) && pair_p(b)) {
		if (length(a) != length(b))
			return 0;
		while (a != NIL && b != NIL) {
			if (!match(car(a), car(b)))
				return 0;
			a = cdr(a);
			b = cdr(b);
		}
		return 1;
	}
	if (dictionary_p(a) && dictionary_p(b)) {
		return dict_contains(a, b) && dict_contains(b, a);
	}
	return 0;
}

static cell dict_lookup(cell d, cell k) {
	cell	x, *v;
	int	h;

	if (0 == dict_len(d))
		return NIL;
	h = hash(k, dict_len(d));
	v = dict_table(d);
	x = v[h];
	while (x != NIL) {
		if (match(caar(x), k))
			return car(x);
		x = cdr(x);
	}
	return UNDEFINED;
}

static cell resize_dict(cell old_d, int k) {
	int	old_k, i, h;
	cell	d, e, n;

	save(old_d);
	if (k < MIN_DICT_LEN)
		k = MIN_DICT_LEN;
	d = make_dict(k);
	k = dict_len(d);
	save(d);
	old_k = dict_len(old_d);
	for (i = 0; i < old_k; i++) {
		for (e = dict_table(old_d)[i]; e != NIL; e = cdr(e)) {
			h = hash(caar(e), k);
			n = cons(car(e), dict_table(d)[h]);
			dict_table(d)[h] = n;
		}
	}
	dict_size(d) = dict_size(old_d);
	d = unsave(1);
	unsave(1);
	return d;
}

static cell grow_dict(cell d) {
	cell	n;

	n = resize_dict(d, (dict_len(d) + 1) * 2 - 1);
	dict_data(d) = dict_data(n);
	return n;
}

static cell copy_dict(cell d) {
	/* see hashsize() about the -1 */
	return resize_dict(d, dict_len(d)-1);
}

static cell dict_add(cell d, cell k, cell v) {
	cell	x, e;
	int	h;

	if (dictionary_p(k))
		return error("bad dictionary key", k);
	Tmp = k;
	save(v);
	save(k);
	Tmp = NIL;
	x = dict_lookup(d, k);
	if (x != UNDEFINED) {
		cadr(x) = v;
		unsave(2);
		return d;
	}
	if (dict_size(d) >= dict_len(d))
		d = grow_dict(d);
	save(d);
	h = hash(k, dict_len(d));
	e = cons(v, NIL);
	e = cons(k, e);
	e = cons(e, dict_table(d)[h]);
	dict_table(d)[h] = e;
	dict_size(d)++;
	unsave(3);
	return d;
}

static cell dict_remove(cell d, cell k) {
	cell	*x, *v;
	int	h;

	if (0 == dict_len(d))
		return NIL;
	h = hash(k, dict_len(d));
	v = dict_table(d);
	x = &v[h];
	while (*x != NIL) {
		if (match(caar(*x), k)) {
			*x = cdr(*x);
			dict_size(d)--;
			break;
		}
		x = &cdr(*x);
	}
	return d;
}

static cell dict_to_list(cell x) {
	cell	n, p, new, last;
	int	i, k;

	n = cons(NIL, NIL);
	save(n);
	k = dict_len(x);
	last = NIL;
	for (i = 0; i < k; i++) {
		for (p = dict_table(x)[i]; p != NIL; p = cdr(p)) {
			car(n) = car(p);
			new = cons(NIL, NIL);
			cdr(n) = new;
			last = n;
			n = cdr(n);
		}
	}
	n = unsave(1);
	if (NIL == last)
		return NIL;
	cdr(last) = NIL;
	return n;
}

static int dict_contains(cell x, cell y) {
	cell	p, t;

	p = dict_to_list(x);
	save(p);
	while (p != NIL) {
		t = dict_lookup(y, caar(p));
		if (UNDEFINED == t || match(cadar(p), cadr(t)) == 0) {
			unsave(1);
			return 0;
		}
		p = cdr(p);
	}
	unsave(1);
	return 1;
}

/*
 * Backend Routines
 */

static int intvalue(cell x) {
	int	of;

	return bignum_to_int(x, &of);
}

static cell flatcopy(cell x, cell *p) {
	cell	n, m;

	n = cons(NIL, NIL);
	save(n);
	while (x != NIL) {
		car(n) = car(x);
		x = cdr(x);
		if (x != NIL) {
			m = cons(NIL, NIL);
			cdr(n) = m;
			n = cdr(n);
		}
	}
	*p = n;
	return unsave(1);
}

static cell rev(cell x) {
	cell	n = NIL;

	for (; x != NIL; x = cdr(x))
		n = cons(car(x), n);
	return n;
}

static cell revb(cell n) {
	cell    m, h;

	if (NIL == n)
		return NIL;
	m = NIL;
	while (n != NIL) {
		h = cdr(n);
		cdr(n) = m;
		m = n;
		n = h;
	}
	return m;
}

static cell rev_string(cell x) {
	cell	s;
	char	*src, *dst;
	int	i, k;

	k = string_len(x)-1;
	s = make_string("", k);
	src = string(x);
	dst = &string(s)[k];
	*dst-- = 0;
	for (i = 0; i < k; i++)
		*dst-- = *src++;
	return s;
}

static cell append(cell a, cell b) {
	cell	n, m;

	if (NIL == a)
		return b;
	if (NIL == b)
		return a;
	n = flatcopy(a, &m);
	cdr(m) = b;
	return n;
}

static cell amend(cell x, cell y) {
	cell	n, v, new, p = NIL;
	int	i, k;

	if (NIL == x || NIL == cdr(x))
		return y;
	n = cons(NIL, NIL);
	save(n);
	v = car(x);
	x = cdr(x);
	i = 0;
	k = -1;
	if (NIL != x) {
		if (!integer_p(car(x))) {
			unsave(1);
			return error("amend: expected integer, got", car(x));
		}
		k = intvalue(car(x));
		x = cdr(x);
	}
	while (y != NIL) {
		if (i == k) {
			car(n) = v;
			if (x != NIL) {
				if (!integer_p(car(x))) {
					unsave(1);
					return
					 error("amend: expected integer, got",
						car(x));
				}
				k = intvalue(car(x));
				x = cdr(x);
			}
		}
		else {
			car(n) = car(y);
		}
		y = cdr(y);
		p = n;
		new = cons(NIL, NIL);
		cdr(n) = new;
		n = cdr(n);
		i++;
	}
	if (k >= i || x != NIL)
		error("amend: range error", make_integer(k));
	n = unsave(1);
	if (p != NIL)
		cdr(p) = NIL;
	else
		n = NIL;
	return n;
}

static cell amend_substring(cell x, cell y) {
	cell	s, v, p;
	int	i, k, kv, r;

	v = car(x);
	x = cdr(x);
	r = 0;
	k = string_len(y)-1;
	kv = string_len(v)-1;
	for (p=x; p != NIL; p = cdr(p)) {
		if (!integer_p(car(p))) {
			unsave(1);
			return error("amend: expected integer, got", car(p));
		}
		i = intvalue(car(p));
		if (i > r) r = i;
		if (i < 0 || r > k)
			return error("amend: range error", car(p));
	}
 	k = (kv+r)>k? kv+r: k;
	s = make_string("", k);
	if (k == 0)
		return s;
	memcpy(string(s), string(y), k);
	save(s);
	while (x != NIL) {
		i = intvalue(car(x));
		memcpy(&string(s)[i], string(v), kv);
		x = cdr(x);
	}
	return unsave(1);
}

static cell amend_string(cell x, cell y) {
	cell	s, v, c;
	int	i, k;

	if (NIL == x || NIL == cdr(x))
		return y;
	v = car(x);
	if (string_p(v))
		return amend_substring(x, y);
	if (!char_p(v))
		return error("amend: expected char, got", v);
	c = char_value(v);
	s = copy_string(y);
	save(s);
	k = string_len(s)-1;
	x = cdr(x);
	while (x != NIL) {
		if (!integer_p(car(x))) {
			unsave(1);
			return error("amend: expected integer, got", car(x));
		}
		i = intvalue(car(x));
		if (i < 0 || i >= k) {
			unsave(1);
			return error("amend: range error", car(x));
		}
		string(s)[i] = c;
		x = cdr(x);
	}
	return unsave(1);
}

static cell amend_in_depth(cell x, cell y, cell v) {
	cell	n, new, p = NIL;
	int	i, k;

	if (NIL == x) {
		if (!atom_p(y))
			return error("amend-in-depth: shape error", y);
		return v;
	}
	if (atom_p(y))
		return error("amend-in-depth: shape error", x);
	n = cons(NIL, NIL);
	save(n);
	i = 0;
	if (!integer_p(car(x))) {
		unsave(1);
		return error("amend-in-depth: expected integer, got", car(x));
	}
	k = intvalue(car(x));
	while (y != NIL) {
		if (i == k) {
			new = amend_in_depth(cdr(x), car(y), v);
			car(n) = new;
		}
		else {
			car(n) = car(y);
		}
		y = cdr(y);
		p = n;
		new = cons(NIL, NIL);
		cdr(n) = new;
		n = cdr(n);
		i++;
	}
	if (k >= i)
		error("amend-in-depth: range error", make_integer(k));
	if (p != NIL)
		cdr(p) = NIL;
	return unsave(1);
}

static cell cut(cell x, cell y) {
	cell	seg, segs, offs, new, cutoff = NIL;
	int	k, i, n, orig;

	if (NIL == x)
		return cons(y, NIL);
	if (integer_p(x))
		x = cons(x, NIL);
	orig = y;
	save(x);
	offs = x;
	segs = cons(NIL, NIL);
	save(segs);
	i = 0;
	while (x != NIL) {
		if (!integer_p(car(x))) {
			unsave(2);
			return error("cut: expected integer, got", car(x));
		}
		k = intvalue(car(x));
		x = cdr(x);
		seg = cons(NIL, NIL);
		save(seg);
		if (i > k)
			return error("cut: range error", offs);
		for (n = 0; i < k; i++, n++) {
			if (NIL == y) {
				unsave(3);
				return error("cut: length error", offs);
			}
			car(seg) = car(y);
			if (i < k-1) {
				new = cons(NIL, NIL);
				cdr(seg) = new;
				seg = cdr(seg);
			}
			y = cdr(y);
		}
		seg = unsave(1);
		if (0 == n)
			seg = NIL;
		car(segs) = seg;
		new = cons(NIL, NIL);
		cutoff = segs;
		cdr(segs) = new;
		segs = cdr(segs);
	}
	if (NIL == orig && cutoff != NIL)
		cdr(cutoff) = NIL;
	else
		car(segs) = y;
	segs = unsave(1);
	unsave(1);
	return segs;
}

static cell cut_string(cell x, cell y) {
	cell	seg, segs, offs, new, orig, cutoff = NIL;
	int	k0, k1, n;

	if (NIL == x)
		return cons(y, NIL);
	if (integer_p(x))
		x = cons(x, NIL);
	orig = y;
	save(x);
	offs = x;
	segs = cons(NIL, NIL);
	save(segs);
	k0 = 0;
	k1 = 0; /*LINT*/
	n = string_len(y);
	while (x != NIL) {
		if (!integer_p(car(x))) {
			unsave(2);
			return error("cut: expected integer, got", car(x));
		}
		k1 = intvalue(car(x));
		x = cdr(x);
		if (k0 > k1 || k1 >= n)
			return error("cut: range error", offs);
		seg = make_string("", k1-k0);
		memcpy(string(seg), string(y)+k0, k1-k0+1);
		string(seg)[k1-k0] = 0;
		car(segs) = seg;
		new = cons(NIL, NIL);
		cutoff = segs;
		cdr(segs) = new;
		segs = cdr(segs);
		k0 = k1;
	}
	if (string_len(orig) < 2 && cutoff != NIL) {
		cdr(cutoff) = NIL;
	}
	else {
		n -= k1;
		new = make_string("", n-1);
		car(segs) = new;
		memcpy(string(car(segs)), string(y)+k1, n);
	}
	segs = unsave(1);
	unsave(1);
	return segs;
}

#define empty_p(x) \
	(NIL == x || (string_p(x) && string_len(x) < 2))

static int false_p(cell x) {
	return  NIL == x ||
		Zero == x ||
		(number_p(x) && real_zero_p(x)) ||
		empty_p(x);
}

static cell drop(cell x, cell y) {
	int	k, r = 0;

	k = intvalue(y);
	if (k < 0) {
		x = rev(x);
		k = -k;
		r = 1;
	}
	save(x);
	while (x != NIL && k--)
		x = cdr(x);
	if (r)
		x = rev(x);
	unsave(1);
	return x;
}

static cell drop_string(cell x, cell y) {
	int	k, n;
	cell	new;

	n = string_len(x)-1;
	k = intvalue(y);
	if (0 == k)
		return x;
	if (k < 0) {
		k = -k;
		if (k > n)
			return make_string("", 0);
		new = make_string("", n - k);
		memcpy(string(new), string(x), n-k+1);
		string(new)[n-k] = 0;
		return new;
	}
	if (k > n)
		return make_string("", 0);
	new = make_string("", n - k);
	memcpy(string(new), string(x)+k, n-k+1);
	return new;
}

static cell expand(cell x) {
	cell	n, new;
	int	k, pos, last = NIL;

	if (NIL == x || (number_p(x) && real_zero_p(x)))
		return NIL;
	if (integer_p(x))
		x = cons(x, NIL);
	save(x);
	n = cons(NIL, NIL);
	save(n);
	pos = 0;
	while (x != NIL) {
		if (!integer_p(car(x)))
			return error("expand: expected integer, got", car(x));
		k = intvalue(car(x));
		if (k < 0)
			return error("expand: range error", car(x));
		while (k) {
			new = make_integer(pos);
			car(n) = new;
			if (k > 0 || cdr(x) != NIL) {
				new = cons(NIL, NIL);
				cdr(n) = new;
				last = n;
				n = cdr(n);
			}
			k--;
		}
		x = cdr(x);
		pos++;
	}
	n = unsave(1);
	unsave(1);
	if (NIL == last)
		return NIL;
	if (last != NIL && NIL == cadr(last))
		cdr(last) = NIL;
	return n;
}

static cell find(cell x, cell y) {
	cell	n, p, new;
	int	k = 0;

	p = NIL;
	n = cons(NIL, NIL);
	save(n);
	while (y != NIL) {
		if (match(x, car(y))) {
			new = make_integer(k);
			car(n) = new;
			new = cons(NIL, NIL);
			cdr(n) = new;
			p = n;
			n = cdr(n);
		}
		y = cdr(y);
		k++;
	}
	n = unsave(1);
	if (p != NIL) {
		cdr(p) = NIL;
		return n;
	}
	else {
		return NIL;
	}
}

static cell find_string(cell x, cell y) {
	cell	n, p, new;
	int	i, k, c;

	c = char_value(x);
	k = string_len(y);
	p = NIL;
	n = cons(NIL, NIL);
	save(n);
	for (i = 0; i < k; i++) {
		if (string(y)[i] == c) {
			new = make_integer(i);
			car(n) = new;
			new = cons(NIL, NIL);
			cdr(n) = new;
			p = n;
			n = cdr(n);
		}
	}
	n = unsave(1);
	if (p != NIL) {
		cdr(p) = NIL;
		return n;
	}
	else {
		return NIL;
	}
}

static cell find_substring(cell x, cell y) {
	cell	n, p, new;
	int	i, k, ks;

	ks = string_len(x)-1;
	k = string_len(y)-1;
	p = NIL;
	n = cons(NIL, NIL);
	save(n);
	for (i = 0; i <= k-ks; i++) {
		if (memcmp(&string(y)[i], string(x), ks) == 0) {
			new = make_integer(i);
			car(n) = new;
			new = cons(NIL, NIL);
			cdr(n) = new;
			p = n;
			n = cdr(n);
		}
	}
	n = unsave(1);
	if (p != NIL) {
		cdr(p) = NIL;
		return n;
	}
	else {
		return NIL;
	}
}

static cell list_to_vector(cell m) {
	cell	n, vec;
	int	k;
	cell	*p;

	k = 0;
	for (n = m; n != NIL; n = cdr(n))
		k++;
	vec = new_vec(T_VECTOR, k*sizeof(cell));
	p = vector(vec);
	for (n = m; n != NIL; n = cdr(n)) {
		*p = car(n);
		p++;
	}
	return vec;
}

static cell string_to_vector(cell s) {
	cell	v, new;
	int	k, i;

	k = string_len(s)-1;
	v = new_vec(T_VECTOR, k*sizeof(cell));
	save(v);
	for (i = 0; i < k; i++) {
		new = make_char(string(s)[i]);
		vector(v)[i] = new;
	}
	unsave(1);
	return v;
}

static cell ndxvec_to_list(cell x) {
	cell	n, new;
	int	k, i;

	k = vector_len(x);
	if (0 == k)
		return NIL;
	n = cons(NIL, NIL);
	save(n);
	for (i=0; i<k; i++) {
		new = make_integer(vector(x)[i]);
		car(n) = new;
		if (i < k-1) {
			new = cons(NIL, NIL);
			cdr(n) = new;
			n = cdr(n);
		}
	}
	n = unsave(1);
	return n;
}

/*
 * Avoid gap sizes of 2^n in shellsort.
 * Hopefully the compiler will optimize the switch.
 * Worst case for shellsort with fixgap() is Theta(n^1.5),
 * meaning
 * 1,000,000,000 steps to sort one million elements
 * instead of
 * 1,000,000,000,000 steps (Theta(n^2)).
 */

static int fixgap(int k) {
	switch (k) {
	case 4:
	case 8:
	case 16:
	case 32:
	case 64:
	case 128:
	case 256:
	case 512:
	case 1024:
	case 2048:
	case 4096:
	case 8192:
	case 16384:
	case 32768:
	case 65536:
	case 131072:
	case 262144:
	case 524288:
	case 1048576:
	case 2097152:
	case 4194304:
	case 8388608:
	case 16777216:
	case 33554432:
	case 67108864:
	case 134217728:
	case 268435456:
	case 536870912:
	case 1073741824:return k-1;
	default:	return k;
	}
}

static void shellsort(cell vals, cell ndxs, int count, int (*p)(cell, cell)) {
	int	gap, i, j;
	cell	tmp, *vv, *nv;

	for (gap = fixgap(count/2); gap > 0; gap = fixgap(gap / 2)) {
		for (i = gap; i < count; i++) {
			for (j = i-gap; j >= 0; j -= gap) {
				vv = vector(vals);
				nv = vector(ndxs);
				if (p(vv[nv[j]], vv[nv[j+gap]]))
					break;
				tmp = nv[j];
				nv[j] = nv[j+gap];
				nv[j+gap] = tmp;
			}
		}
	}
}

static cell grade(int (*p)(cell, cell), cell x) {
	cell	vals, ndxs, *v;
	int	i, k;

	vals = list_to_vector(x);
	save(vals);
	k = vector_len(vals);
	/*
	 * allocate vector as string, so it does not get scanned during GC
	 */
	ndxs = new_vec(T_STRING, k * sizeof(cell));
	save(ndxs);
	v = vector(ndxs);
	for (i = 0; i < k; i++)
		v[i] = i;
	shellsort(vals, ndxs, k, p);
	ndxs = car(Stack);
	ndxs = ndxvec_to_list(ndxs);
	unsave(2);
	return ndxs;
}

static cell string_to_list(cell s, cell xnil) {
	cell	n, new;
	int	k, i;

	k = string_len(s) - 1;
	if (k < 1)
		return NIL;
	n = cons(NIL, xnil);
	save(n);
	for (i = 0; i < k; i++) {
		new = make_char(string(s)[i]);
		car(n) = new;
		if (i+1 < k) {
			new = cons(NIL, xnil);
			cdr(n) = new;
			n = cdr(n);
		}
	}
	return unsave(1);
}

static cell list_to_string(cell x) {
	cell	p, n;
	int	k;
	char	*s;

	if (NIL == x)
		return make_string("", 0);
	if (atom_p(x))
		return x;
	k = 0;
	for (p=x; p != NIL; p = cdr(p)) {
		if (!char_p(car(p)))
			return x;
		k++;
	}
	n = make_string("", k);
	s = string(n);
	for (p=x; p != NIL; p = cdr(p))
		*s++ = char_value(car(p));
	*s = 0;
	return n;
}

static void conv_to_strlst(cell x) {
	cell	n;

	while (x != NIL) {
		n = list_to_string(car(x));
		car(x) = n;
		x = cdr(x);
	}
}

static cell group(cell x) {
	cell	n, f, p, g, new, ht;
	int	i;

	if (NIL == x)
		return NIL;
	ht = make_dict(length(x));
	save(ht);
	for (i = 0, n = x; n != NIL; n = cdr(n), i++) {
		p = dict_lookup(ht, car(n));
		p = cons(make_integer(i), atom_p(p)? NIL: cadr(p));
		ht = dict_add(ht, car(n), p);
		car(Stack) = ht;
		if (undefined_p(ht)) {
			unsave(1);
			return UNDEFINED;
		}
	}
	g = cons(NIL, NIL);
	save(g);
	p = g;
	for (n = x; n != NIL; n = cdr(n)) {
		f = dict_lookup(ht, car(n));
		if (f != UNDEFINED) {
			car(g) = revb(cadr(f));
			new = cons(NIL, NIL);
			cdr(g) = new;
			p = g;
			g = cdr(g);
			dict_remove(ht, car(n));
		}
	}
	cdr(p) = NIL;
	x = unsave(1);
	unsave(1);
	return x;
}

static cell group_string(cell x) {
	cell	n;

	n = string_to_list(x, NIL);
	save(n);
	n = group(n);
	unsave(1);
	return n;
}

static cell join(cell y, cell x) {
	cell	n, p;

	if (list_p(x)) {
		if (list_p(y))
			x = append(y, x);
		else
			x = cons(y, x);
		return x;
	}
	if (list_p(y)) {
		if (NIL == y) {
			x = cons(x, NIL);
		}
		else {
			y = flatcopy(y, &p);
			save(y);
			x = cons(x, NIL);
			cdr(p) = x;
			unsave(1);
			x = y;
		}
		return x;
	}
	if (char_p(x) && char_p(y)) {
		n = make_string("", 2);
		string(n)[1] = char_value(x);
		string(n)[0] = char_value(y);
		return n;
	}
	if (string_p(x) && char_p(y)) {
		n = make_string("", string_len(x));
		memcpy(string(n) + 1, string(x), string_len(x));
		string(n)[0] = intvalue(y);
		return n;
	}
	if (char_p(x) && string_p(y)) {
		n = make_string("", string_len(y));
		memcpy(string(n), string(y), string_len(y));
		string(n)[string_len(y)-1] = intvalue(x);
		return n;
	}
	if (string_p(x) && string_p(y)) {
		n = make_string("", string_len(x) + string_len(y)-2);
		memcpy(string(n), string(y), string_len(y));
		memcpy(string(n) + string_len(y)-1, string(x), string_len(x));
		return n;
	}
	n = cons(x, NIL);
	n = cons(y, n);
	return n;
}

static int less(cell x, cell y) {
	if (number_p(x)) {
		if (!number_p(y))
			return error("grade: expected number, got", y);
		return real_less_p(x, y);
	}
	if (symbol_p(x)) {
		if (!symbol_p(y))
			return error("grade: expected symbol, got", y);
		return strcmp(symbol_name(x), symbol_name(y)) < 0;
	}
	if (char_p(x)) {
		if (!char_p(y))
			return error("grade: expected char, got", y);
		return char_value(x) < char_value(y);
	}
	if (string_p(x)) {
		if (!string_p(y))
			return error("grade: expected string, got", y);
		return strcmp(string(x), string(y)) < 0;
	}
	if (!list_p(x))
		return error("grade: expected list, got", x);
	if (!list_p(y))
		return error("grade: expected list, got", y);
	while (x != NIL && y != NIL) {
		if (less(car(x), car(y)))
			return 1;
		if (less(car(y), car(x)))
			return 0;
		x = cdr(x);
		y = cdr(y);
	}
	return y != NIL;
}

static int more(cell x, cell y) {
	return less(y, x);
}

static cell ndx(cell x, int k, cell y) {
	int	i;
	cell	n, new;
	cell	*v;

	if (integer_p(y)) {
		i = intvalue(y);
		if (i < 0 || i >= k)
			return error("index: range error", y);
		v = vector(x);
		return v[i];
	}
	if (NIL == y)
		return NIL;
	if (atom_p(y))
		return error("index: expected integer, got", y);
	n = cons(NIL, NIL);
	save(n);
	for (; y != NIL; y = cdr(y)) {
		new = ndx(x, k, car(y));
		car(n) = new;
		if (cdr(y) != NIL) {
			new = cons(NIL, NIL);
			cdr(n) = new;
			n = cdr(n);
		}
	}
	n = unsave(1);
	return n;
}

static cell ndx_in_depth(cell x, cell y) {
	cell	p;
	int	k;

	if (atom_p(y))
		y = cons(y, NIL);
	save(y);
	p = x;
	while (NIL != y) {
		if (!integer_p(car(y))) {
			unsave(1);
			return error("index-in-depth: expected integer, got",
					car(y));
		}
		k = intvalue(car(y));
		if (atom_p(p)) {
			unsave(1);
			return error("index-in-depth: shape error", p);
		}
		while (k > 0 && p != NIL) {
			p = cdr(p);
			k--;
		}
		if (atom_p(p)) {
			unsave(1);
			return error("index-in-depth: shape error", p);
		}
		p = car(p);
		y = cdr(y);
	}
	unsave(1);
	return p;
}

static cell reshape3(cell shape, cell *next, cell src) {
	int	i, k, str = 1;
	cell	n, new;

	if (NIL == shape) {
		if (NIL == *next)
			*next = src;
		n = car(*next);
		*next = cdr(*next);
		return n;
	}
	n = cons(NIL, NIL);
	save(n);
	if (!integer_p(car(shape))) {
		unsave(1);
		return error("reshape: expected integer, got", car(shape));
	}
	k = intvalue(car(shape));
	if (-1 == k) {
		k = length(src)/2;
	}
	if (k < 1) {
		unsave(1);
		return error("reshape: range error", car(shape));
	}
	for (i = 0; i < k; i++) {
		new = reshape3(cdr(shape), next, src);
		if (!char_p(new))
			str = 0;
		car(n) = new;
		if (i < k - 1) {
			new = cons(NIL, NIL);
			cdr(n) = new;
			n = cdr(n);
		}
	}
	n = unsave(1);
	if (str) {
		save(n);
		n = list_to_string(n);
		unsave(1);
	}
	return n;
}

static cell joinstrs(cell x) {
	cell	n, last = NIL, new;

	n = cons(NIL, NIL);
	save(n);
	while (x != NIL) {
		car(n) = car(x);
		if (string_p(car(x))) {
			while (cdr(x) != NIL && string_p(cadr(x))) {
				new = join(car(n), cadr(x));
				car(n) = new;
				x = cdr(x);
			}
		}
		new = cons(NIL, NIL);
		cdr(n) = new;
		last = n;
		n = cdr(n);
		x = cdr(x);
	}
	n = unsave(1);
	if (NIL == last) {
		n = NIL;
	}
	else {
		cdr(last) = NIL;
		if (string_p(car(n)) && NIL == cdr(n))
			n = car(n);
	}
	return n;
}

static cell range(cell x) {
	cell	n, m, p, new;
	int	i, k,str = 0;

	if (list_p(x))
		n = group(x);
	else
		n = group_string(x);
	if (NIL == n)
		return string_p(x)? make_string("", 0): NIL;
	save(n);
	if (!atom_p(n))
		for (p = n; p != NIL; p = cdr(p))
			car(p) = caar(p);
	k = intvalue(car(n));
	if (string_p(x)) {
		x = string_to_list(x, NIL);
		str = 1;
	}
	save(x);
	m = cons(NIL, NIL);
	save(m);
	for (i = 0, p = x; p != NIL; p = cdr(p), i++) {
		if (i == k) {
			car(m) = car(p);
			n = cdr(n);
			if (NIL == n)
				break;
			k = intvalue(car(n));
			new = cons(NIL, NIL);
			cdr(m) = new;
			m = cdr(m);
		}
	}
	n = unsave(1);
	if (str) {
		car(Stack) = n;
		n = list_to_string(n);
	}
	unsave(2);
	return n;
}

static cell reshape(cell x, cell y) {
	if (atom_p(x))
		x = cons(x, NIL);
	save(x);
	car(Stack) = x;
	x = joinstrs(x);
	car(Stack) = x;
	if (string_p(x)) {
		x = string_to_list(x, NIL);
		car(Stack) = x;
	}
	if (atom_p(y))
		y = cons(y, NIL);
	save(y);
	x = reshape3(y, &x, x);
	unsave(2);
	return x;
}

static cell take(cell x, int k) {
	cell	a, n, m;
	int	r = 0;

	if (NIL == x || 0 == k)
		return NIL;
	a = x;
	if (k < 0) {
		a = x = rev(x);
		k = -k;
		r = 1;
	}
	save(x);
	n = cons(NIL, NIL);
	save(n);
	while (k--) {
		if (NIL == x)
			x = a;
		car(n) = car(x);
		x = cdr(x);
		if (k) {
			m = cons(NIL, NIL);
			cdr(n) = m;
			n = cdr(n);
		}
	}
	x = car(Stack);
	if (r) {
		x = rev(x);
	}
	unsave(2);
	return x;
}

static cell take_string(cell x, int k) {
	cell	n;
	int	len, i, j;

	len = string_len(x) - 1;
	if (len < 1 || 0 == k)
		return make_string("", 0);
	n = make_string("", abs(k));
	if (k > 0) {
		j = 0;
		for (i = 0; i < k; i++) {
			string(n)[i] = string(x)[j];
			if (++j >= len)
				j = 0;
		}
	}
	else {
		k = -k;
		j = len-1;
		for (i = k-1; i >= 0; i--) {
			string(n)[i] = string(x)[j];
			if (--j < 0)
				j = len-1;
		}
	}
	return n;
}

static cell rotate(cell x, cell y) {
	int	k, rot;
	cell	n, m, p;

	if (NIL == x)
		return NIL;
	k = length(x);
	if (k < 2)
		return x;
	rot = intvalue(y) % k;
	if (rot > 0) {
		n = take(x, -rot);
		save(n);
		m = take(x, k-rot);
		for (p = n; cdr(p) != NIL; p = cdr(p))
			;
		cdr(p) = m;
		unsave(1);
		return n;
	}
	else {
		n = take(x, -rot);
		save(n);
		for (p = x; rot++; p = cdr(p))
			;
		p = flatcopy(p, &m);
		cdr(m) = n;
		unsave(1);
		return p;
	}
}

static cell rotate_string(cell x, cell y) {
	int	k, rot;
	cell	n;

	k = string_len(x) - 1;
	if (0 == k)
		return x;
	rot = intvalue(y) % k;
	if (k < 2 || 0 == rot)
		return x;
	if (rot > 0) {
		n = make_string("", k);
		memcpy(string(n), string(x)+k-rot, rot);
		memcpy(string(n)+rot, string(x), k-rot);
		return n;
	}
	else {
		rot = -rot;
		n = make_string("", k);
		memcpy(string(n), string(x)+rot, k-rot);
		memcpy(string(n)+k-rot, string(x), rot);
		return n;
	}
}

static cell eqv_p(cell a, cell b) {
	if (a == b)
		return 1;
	if (number_p(a) && number_p(b))
		return real_equal_p(a, b);
	return 0;
}

static cell common_prefix(cell x, cell y) {
	cell	n, p, new;

	if (atom_p(x) || atom_p(y))
		return NIL;
	p = n = cons(NIL, NIL);
	save(n);
	while (x != NIL && y != NIL) {
		if (!eqv_p(car(x), car(y)))
			break;
		car(n) = car(x);
		p = n;
		new = cons(NIL, NIL);
		cdr(n) = new;
		n = cdr(n);
		x = cdr(x);
		y = cdr(y);
	}
	cdr(p) = NIL;
	n = unsave(1);
	return NIL == car(n)? NIL: n;
}

static cell shape(cell x) {
	cell	s, s2, p, new;

	if (string_p(x))
		return cons(make_integer(string_len(x)-1), NIL);
	if (atom_p(x))
		return Zero;
	if (NIL == cdr(x))
		return cons(One, NIL);
	s = shape(car(x));
	save(s);
	for (p = cdr(x); p != NIL; p = cdr(p)) {
		s2 = shape(car(p));
		save(s2);
		new = common_prefix(s, s2);
		cadr(Stack) = new;
		unsave(1);
		s = car(Stack);
	}
	s2 = make_integer(length(x));
	s = unsave(1);
	if (!bignum_equal_p(s2, One))
		s = cons(s2, s);
	return s;
}

static cell split(cell x, cell y) {
	cell	grp;
	cell	n, m, new;
	int	i = -1, k = -1;

	if (NIL == x)
		return NIL;
	if (integer_p(y))
		y = cons(y, NIL);
	save(y);
	grp = y;
	n = cons(NIL, NIL);
	save(n);
	m = cons(NIL, NIL);
	save(m);
	while (x != NIL) {
		if (i >= k) {
			if (!integer_p(car(y))) {
				unsave(3);
				return error("split: expected integer, got",
						car(y));
			}
			if (i >= 0) {
				car(n) = unsave(1);
				m = cons(NIL, NIL);
				save(m);
				new = cons(NIL, NIL);
				cdr(n) = new;
				n = cdr(n);
				y = cdr(y);
				if (NIL == y)
					y = grp;
			}
			k = intvalue(car(y));
			if (k < 1) {
				unsave(3);
				return error("split: range error",
						make_integer(k));
			}
			i = 0;
		}
		car(m) = car(x);
		if (cdr(x) != NIL && i < k-1) {
			new = cons(NIL, NIL);
			cdr(m) = new;
			m = cdr(m);
		}
		x = cdr(x);
		i++;
	}
	car(n) = unsave(1);
	n = unsave(1);
	unsave(1);
	return n;
}

static cell split_string(cell x, cell y) {
	cell	grp;
	cell	n, new;
	int	i, k, len;

	len = string_len(x) - 1;
	if (len < 1)
		return NIL;
	if (integer_p(y))
		y = cons(y, NIL);
	save(y);
	grp = y;
	n = cons(NIL, NIL);
	save(n);
	i = 0;
	while (i < len) {
		if (!integer_p(car(y))) {
			unsave(2);
			return error("split: expected integer, got", car(y));
		}
		k = intvalue(car(y));
		if (k < 1) {
			unsave(3);
			return error("split: range error", car(y));
		}
		y = cdr(y);
		if (NIL == y)
			y = grp;
		if (k > len-i)
			k = len-i;
		new = make_string("", k);
		memcpy(string(new), string(x)+i, k);
		string(new)[k] = 0;
		car(n) = new;
		i += k;
		if (i < len) {
			new = cons(NIL, NIL);
			cdr(n) = new;
			n = cdr(n);
		}
	}
	n = unsave(1);
	unsave(1);
	return n;
}

static int anylist(cell x) {
	cell    p; 

	for (p=x; p != NIL; p = cdr(p))
		if (pair_p(car(p)))
			return 1; 
	return 0;
}

static int anynil(cell x) {
	cell    p; 

	for (p=x; p != NIL; p = cdr(p))
		if (NIL == car(p))
			return 1; 
	return 0;
}

static cell transpose(cell x) {
	cell	n, m, p, q, dummy, new;

	if (atom_p(x) || !anylist(x))
		return x;
	save(flatcopy(x, &dummy));
	n = cons(NIL, NIL);
	save(n);
	for (;;) {
		if (anynil(cadr(Stack)))
			break;
		m = cons(NIL, NIL);
		save(m);
		for (p=caddr(Stack); p != NIL; p = cdr(p)) {
			q = car(p);
			if (atom_p(q)) {
				car(m) = q;
			}
			else {
				car(m) = car(q);
				car(p) = cdar(p);
			}
			if (cdr(p) != NIL) {
				new = cons(NIL, NIL);
				cdr(m) = new;
				m = cdr(m);
			}
		}
		m = unsave(1);
		car(n) = m;
		if (anynil(cadr(Stack)))
			break;
		new = cons(NIL, NIL);
		cdr(n) = new;
		n = cdr(n);
	}
	n = unsave(1);
	if (anylist(unsave(1)))
		error("transpose: shape error", x);
	return n;
}

/*
 * Virtual Machine
 */

static void push(cell x) {
	Dstack = cons(x, Dstack);
}

static cell pop(void) {
	cell	n;

	if (NIL == Dstack)
		return error("stack underflow", VOID);
	n = car(Dstack);
	Dstack = cdr(Dstack);
	return n;
}

static cell	(*F2)(cell x, cell y);
static cell	(*F1)(cell x);
static char	*N, B[100];

static cell rec2(cell x, cell y) {
	cell	n, p, q, new;

	if (atom_p(x) && atom_p(y))
		return (*F2)(x, y);
	n = cons(NIL, NIL);
	save(n);
	if (list_p(x) && list_p(y)) {
		for (p=x, q=y; p != NIL && q != NIL; p = cdr(p), q = cdr(q)) {
			new = rec2(car(p), car(q));
			car(n) = new;
			if (cdr(p) != NIL) {
				new = cons(NIL, NIL);
				cdr(n) = new;
				n = cdr(n);
			}
		}
		if (p != NIL || q != NIL) {
			n = cons(y, NIL);
			n = cons(x, n);
			unsave(1);
			sprintf(B, "%s: shape error", N);
			return error(B, n);
		}
	}
	else if (list_p(x)) {
		for (p=x; p != NIL; p = cdr(p)) {
			new = rec2(car(p), y);
			car(n) = new;
			if (cdr(p) != NIL) {
				new = cons(NIL, NIL);
				cdr(n) = new;
				n = cdr(n);
			}
		}
	}
	else if (list_p(y)) {
		for (p=y; p != NIL; p = cdr(p)) {
			new = rec2(x, car(p));
			car(n) = new;
			if (cdr(p) != NIL) {
				new = cons(NIL, NIL);
				cdr(n) = new;
				n = cdr(n);
			}
		}
	}
	return unsave(1);
}

static void dyadrec(char *s, cell (*f)(cell, cell), cell x, cell y) {
	cell	r;

	N = s;
	F2 = f;
	r = rec2(x, y);
	Dstack = cdr(Dstack);
	car(Dstack) = r;
}

static cell rec1(cell x) {
	cell	n, p, new;

	if (atom_p(x))
		return (*F1)(x);
	n = cons(NIL, NIL);
	save(n);
	for (p=x; p != NIL; p = cdr(p)) {
		new = rec1(car(p));
		car(n) = new;
		if (cdr(p) != NIL) {
			new = cons(NIL, NIL);
			cdr(n) = new;
			n = cdr(n);
		}
	}
	return unsave(1);
}

static void monadrec(char *s, cell (*f)(cell), cell x) {
	cell	r;

	N = s;
	F1 = f;
	r = rec1(x);
	car(Dstack) = r;
}

static void save_vars(cell vars) {
	cell	n, v;

	for (v = vars; v != NIL; v = cdr(v)) {
		if (!variable_p(car(v))) {
			error("non-variable in variable list", vars);
			return;
		}
		n = cons(var_value(car(v)), car(Locals));
		var_value(car(v)) = NO_VALUE;
		car(Locals) = n;
	}
	n = cons(rev(vars), car(Locals));
	car(Locals) = n;
}

static void unsave_vars(void) {
	cell	v, a;

	var_value(S_thisfn) = caar(Locals);
	car(Locals) = cdar(Locals);
	if (car(Locals) != NIL) {
		v = caar(Locals);
		for (a = cdar(Locals); a != NIL; a = cdr(a)) {
			var_value(car(v)) = car(a);
			v = cdr(v);
		}
	}
	Locals = cdr(Locals);
}

#define ONE_ARG(name) \
	if (NIL == Dstack) { \
		error("too few arguments", make_variable(name, NIL)); \
		return; \
	}

#define TWO_ARGS(name) \
	if (NIL == Dstack || NIL == cdr(Dstack)) { \
		error("too few arguments", make_variable(name, NIL)); \
		return; \
	}

#define THREE_ARGS(name) \
	if (NIL == Dstack || NIL == cdr(Dstack) || NIL == cddr(Dstack)) { \
		error("too few arguments", make_variable(name, NIL)); \
		return; \
	}

static void unknown1(char *name) {
	cell	n;
	char	b[100];

	n = cons(car(Dstack), NIL);
	sprintf(b, "%s: type error", name);
	error(b, n);
}

static void unknown2(char *name) {
	cell	n;
	char	b[100];

	n = cons(car(Dstack), NIL);
	n = cons(cadr(Dstack), n);
	sprintf(b, "%s: type error", name);
	error(b, n);
}

static void unknown3(char *name) {
	cell	n;
	char	b[100];

	n = cons(car(Dstack), NIL);
	n = cons(cadr(Dstack), n);
	n = cons(caddr(Dstack), n);
	sprintf(b, "%s: type error", name);
	error(b, n);
}

static void binop(cell (*op)(cell, cell), cell x, cell y) {
	cell	r;

	r = op(y, x);
	pop();
	car(Dstack) = r;
}

/*
 * Operators (primitive functions)
 */

static void op_amend(void) {
	cell	x, y;

	TWO_ARGS("amend")
	y = cadr(Dstack);
	x = car(Dstack);
	if (list_p(y) && list_p(x)) {
		x = amend(x, y);
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	if (string_p(y) && list_p(x)) {
		x = amend_string(x, y);
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	unknown2("amend");
}

static void op_amendd(void) {
	cell	x, y;

	TWO_ARGS("amendd")
	y = cadr(Dstack);
	x = car(Dstack);
	if (pair_p(y) && pair_p(x)) {
		x = amend_in_depth(cdr(x), y, car(x));
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	unknown2("amendd");
}

static void op_apply(void) {
	cell	x, y;
	cell	n, m, f;
	int	na;

	TWO_ARGS("apply")
	y = cadr(Dstack);
	x = car(Dstack);
	if (function_p(y)) {
		if (!list_p(x)) {
			x = cons(x, NIL);
			car(Dstack) = x;
		}
		na = fun_arity(y);
		if (na != length(x)) {
			error("apply: wrong number of arguments", x);
			return;
		}
		if (NIL == x) {
			n = cons(S_pop0, NIL);
			n = cons(S_call0, n);
			n = cons(y, n);
			Dstack = cdr(Dstack);
			car(Dstack) = n;
		}
		else {
			f = flatcopy(x, &m);
			Dstack = cdr(Dstack);
			car(Dstack) = f;
			n = cons(1==na? S_pop1: 2==na? S_pop2: S_pop3, NIL);
			n = cons(1==na? S_call1: 2==na? S_call2: S_call3, n);
			n = cons(y, n);
			cdr(m) = n;
		}
		State = S_APPLY;
		return;
	}
	unknown2("apply");
}

static void op_atom(void) {
	cell	x;

	ONE_ARG("atom");
	x = car(Dstack);
	car(Dstack) = string_p(x) && !empty_p(x)? Zero: atom_p(x)? One: Zero;
}

#define has_locals(x) \
	(list_p(car(x)) && list_p(cdr(x)) && S_clear == cadr(x))

static void call(int arity) {
	cell	x, y, z, f, n;
	char	name[] = "call";

	if (3 == arity) {
		if (	NIL == Dstack ||
			NIL == cdr(Dstack) ||
			NIL == cddr(Dstack) ||
			NIL == cdddr(Dstack)
		) {
			error("too few arguments", make_variable(name, NIL));
			return;
		}
		y = cdr(Dstack);
		z = cdddr(Dstack);
		x = car(z);
		car(z) = car(y);
		car(y) = x;
	}
	else if (2 == arity) {
		THREE_ARGS(name);
		y = cdr(Dstack);
		z = cddr(Dstack);
		x = car(z);
		car(z) = car(y);
		car(y) = x;
	}
	else if (1 == arity) {
		TWO_ARGS(name);
	}
	else {
		ONE_ARG(name);
	}
	x = car(Dstack);
	if (function_p(x)) {
		if (fun_arity(x) != arity) {
			error("wrong arity", x);
			return;
		}
		Locals = cons(NIL, Locals);
		f = x;
		x = fun_body(x);
		if (has_locals(x)) {
			save_vars(car(x));
			x = cddr(x);
		}
		n = cons(var_value(S_thisfn), car(Locals));
		car(Locals) = n;
		var_value(S_thisfn) = f;
		car(Dstack) = x;
		State = S_APPLY;
		return;
	}
	unknown1(name);
}

static void op_call0(void) { call(0); }
static void op_call1(void) { call(1); }
static void op_call2(void) { call(2); }
static void op_call3(void) { call(3); }

static cell safe_char(cell x) {
	int	k;

	if (integer_p(x)) {
		k = intvalue(x);
		if (k < 0 || k > 255)
			return error("char: domain error", x);
		return make_char(k);
	}
	return error("char: type error", cons(x, NIL));
}

static void op_char(void) {
	cell	x;

	ONE_ARG("char");
	x = car(Dstack);
	monadrec("char", safe_char, x);
}

static void op_clear(void) {
	pop();
}

static void conv(int s) {
	cell	x, y, n;

	TWO_ARGS("converge")
	y = cadr(Dstack);
	x = car(Dstack);
	if (function_p(x)) {
		if (S_CONV == s)
			push(y);
		else
			push(cons(y, NIL));
		push(Barrier);
		n = cons(x, NIL);
		push(cons(y, n));
		State = s;
		return;
	}
	unknown2("converge");
}

static void op_conv(void) {
	conv(S_CONV);
}

static void op_cut(void) {
	cell	x, y;

	TWO_ARGS("cut")
	y = cadr(Dstack);
	x = car(Dstack);
	if ((list_p(y) || integer_p(y)) && list_p(x)) {
		binop(cut, x, y);
		return;
	}
	if ((list_p(y) || integer_p(y)) && string_p(x)) {
		binop(cut_string, x, y);
		return;
	}
	unknown2("cut");
}

static void op_def(void) {
	cell	x, y, n;
	char	name[TOKEN_LENGTH+1];

	TWO_ARGS("define")
	y = cadr(Dstack);
	x = car(Dstack);
	if (symbol_p(y)) {
		y = make_variable(symbol_name(y), NIL);
		var_value(y) = x;
		if (Module != UNDEFINED) {
			Module = cons(y, Module);
			strcpy(name, var_name(y));
			mkglobal(name);
			n = make_variable(name, NIL);
			var_value(n) = var_value(y);
		}
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	unknown2("define");
}

static cell safe_div(cell x, cell y) {
	if (number_p(x) && number_p(y)) {
		return real_divide(x, y);
	}
	return error("divide: type error", cons(x, cons(y, NIL)));
}

static void op_div(void) {
	cell	x, y;

	TWO_ARGS("divide")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("divide", safe_div, y, x);
}

static void op_down(void) {
	cell	x;

	ONE_ARG("grade-down");
	x = car(Dstack);
	if (list_p(x)) {
		x = grade(more, x);
		car(Dstack) = x;
		return;
	}
	if (string_p(x)) {
		x = string_to_list(x, NIL);
		car(Dstack) = x;
		x = grade(more, x);
		car(Dstack) = x;
		return;
	}
	unknown1("grade-down");
}

static void op_drop(void) {
	cell	x, y;

	TWO_ARGS("drop")
	y = cadr(Dstack);
	x = car(Dstack);
	if (integer_p(y) && list_p(x)) {
		binop(drop, y, x);
		return;
	}
	if (integer_p(y) && string_p(x)) {
		binop(drop_string, y, x);
		return;
	}
	if (dictionary_p(x)) {
		x = dict_remove(x, y);
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	unknown2("drop");
}

static void op_each(void) {
	cell	x, y, n;

	TWO_ARGS("each")
	y = cadr(Dstack);
	x = car(Dstack);
	if ((list_p(y) || string_p(y)) && function_p(x)) {
		if (NIL == y || empty_p(y)) {
			Dstack = cdr(Dstack);
			car(Dstack) = y;
		}
		else {
			push(NIL);
			push(Barrier);
			n = cons(x, NIL);
			if (string_p(y)) {
				save(n);
				push(cons(make_char(string(y)[0]), n));
				unsave(1);
				y = string_to_list(y, STRING_NIL);
			}
			else {
				push(cons(car(y), n));
			}
			cadr(cdddr(Dstack)) = cdr(y);
			State = S_EACH;
		}
		return;
	}
	if (dictionary_p(y) && function_p(x)) {
		x = dict_to_list(y);
		cadr(Dstack) = x;
		op_each();
		return;
	}
	if (function_p(x)) {
		n = cons(x, NIL);
		n = cons(y, n);
		Dstack = cdr(Dstack);
		car(Dstack) = n;
		State = S_APPLY;
		return;
	}
	unknown2("each");
}

static void op_each2(void) {
	cell	x, y, z, n;

	THREE_ARGS("each-2")
	z = caddr(Dstack);
	y = cadr(Dstack);
	x = car(Dstack);
	if (	(list_p(z) || string_p(z)) &&
		(list_p(y) || string_p(y)) &&
		function_p(x)
	) {
		if (empty_p(y) || empty_p(z)) {
			Dstack = cddr(Dstack);
			if (NIL == y || NIL == z)
				x = NIL;
			else
				x = make_string("", 0);
			car(Dstack) = x;
		}
		else {
			if (string_p(y)) {
				y = string_to_list(y, STRING_NIL);
				cadr(Dstack) = y;
			}
			if (string_p(z)) {
				z = string_to_list(z, STRING_NIL);
				caddr(Dstack) = z;
			}
			push(NIL);
			push(Barrier);
			n = cons(x, NIL);
			n = cons(car(y), n);
			push(cons(car(z), n));
			cadr(cdddr(Dstack)) = cdr(y);
			caddr(cdddr(Dstack)) = cdr(z);
			State = S_EACH2;
		}
		return;
	}
	if (function_p(x)) {
		n = cons(x, NIL);
		n = cons(y, n);
		n = cons(z, n);
		Dstack = cddr(Dstack);
		car(Dstack) = n;
		State = S_APPLY;
		return;
	}
	unknown3("each-2");
}

static void op_eachl(void) {
	cell	x, y, z, n;

	THREE_ARGS("each-left")
	z = caddr(Dstack);
	y = cadr(Dstack);
	x = car(Dstack);
	if ((list_p(y) || string_p(y)) && function_p(x)) {
		if (NIL == y || empty_p(y)) {
			Dstack = cddr(Dstack);
			car(Dstack) = y;
		}
		else {
			if (string_p(y)) {
				y = string_to_list(y, STRING_NIL);
				cadr(Dstack) = y;
			}
			push(NIL);
			push(Barrier);
			n = cons(x, NIL);
			n = cons(car(y), n);
			push(cons(z, n));
			cadr(cdddr(Dstack)) = cdr(y);
			State = S_EACHL;
		}
		return;
	}
	if (function_p(x)) {
		n = cons(x, NIL);
		n = cons(y, n);
		n = cons(z, n);
		Dstack = cddr(Dstack);
		car(Dstack) = n;
		State = S_APPLY;
		return;
	}
	unknown3("each-left");
}

static void op_eachp(void) {
	cell	x, y, n;

	TWO_ARGS("each-pair")
	y = cadr(Dstack);
	x = car(Dstack);
	if ((list_p(y) || string_p(y)) && function_p(x)) {
		if (	NIL == y || NIL == cdr(y) ||
			empty_p(y) || (string_p(y) && string_len(y) < 3)
		) {
			Dstack = cdr(Dstack);
			car(Dstack) = y;
		}
		else {
			if (string_p(y)) {
				y = string_to_list(y, STRING_NIL);
				cadr(Dstack) = y;
			}
			push(NIL);
			push(Barrier);
			n = cons(x, NIL);
			n = cons(cadr(y), n);
			push(cons(car(y), n));
			cadr(cdddr(Dstack)) = cdr(y);
			State = S_EACHP;
		}
		return;
	}
	Dstack = cdr(Dstack);
	car(Dstack) = y;
}

static void op_eachr(void) {
	cell	x, y, z, n;

	THREE_ARGS("each-right")
	z = caddr(Dstack);
	y = cadr(Dstack);
	x = car(Dstack);
	if ((list_p(y) || string_p(y)) && function_p(x)) {
		if (NIL == y || empty_p(y)) {
			Dstack = cddr(Dstack);
			car(Dstack) = y;
		}
		else {
			if (string_p(y)) {
				y = string_to_list(y, STRING_NIL);
				cadr(Dstack) = y;
			}
			push(NIL);
			push(Barrier);
			n = cons(x, NIL);
			n = cons(z, n);
			push(cons(car(y), n));
			cadr(cdddr(Dstack)) = cdr(y);
			State = S_EACHR;
		}
		return;
	}
	if (function_p(x)) {
		n = cons(x, NIL);
		n = cons(z, n);
		n = cons(y, n);
		Dstack = cddr(Dstack);
		car(Dstack) = n;
		State = S_APPLY;
		return;
	}
	unknown3("each-right");
}

static void op_enum(void) {
	cell	x, n, new;
	int	i, k;

	ONE_ARG("enumerate");
	x = car(Dstack);
	if (integer_p(x)) {
		k = intvalue(x);
		if (k < 0) {
			error("enumerate: domain error", x);
			return;
		}
		if (0 == k) {
			car(Dstack) = NIL;
			return;
		}
		n = cons(NIL, NIL);
		save(n);
		for (i = 0; i < k; i++) {
			new = make_integer(i);
			car(n) = new;
			if (i < k-1) {
				new = cons(NIL, NIL);
				cdr(n) = new;
				n = cdr(n);
			}
		}
		car(Dstack) = unsave(1);
		return;
	}
	unknown1("enumerate");
}

static int compare(char *name,
	    int (*ncmp)(cell a, cell b),
	    int (*ccmp)(cell a, cell b),
	    int (*scmp)(char *s1, char *s2, int k1, int k2),
	    cell x, cell y
) {
	char	*sx, *sy;
	int	kx, ky;

	if (number_p(y) && number_p(x))
		return ncmp(y, x);
	if (char_p(y) && char_p(x))
		return ccmp(y, x);
	if (variable_p(x))
		return compare(name, ncmp, ccmp, scmp, var_symbol(x), y);
	if (variable_p(y))
		return compare(name, ncmp, ccmp, scmp, x, var_symbol(y));
	if ((string_p(x) && string_p(y)) || (symbol_p(x) && symbol_p(y))) {
		sx = string_p(x)? string(x): symbol_name(x);
		sy = string_p(y)? string(y): symbol_name(y);
		kx = string_p(x)? string_len(x): symbol_len(x);
		ky = string_p(y)? string_len(y): symbol_len(y);
		return scmp(sy, sx, ky, kx);
	}
	unknown2(name);
	return 0;
}

static int str_equal_p(char *s1, char *s2, int k1, int k2) {
	return k1 == k2 && memcmp(s1, s2, k1) == 0;
}

static int str_less_p(char *s1, char *s2, int k1, int k2) {
	int	k = k1 < k2? k1: k2;

	return memcmp(s1, s2, k) < 0;
}

static int chr_equal_p(cell x, cell y) {
	return char_value(x) == char_value(y);
}

static int chr_less_p(cell x, cell y) {
	return char_value(x) < char_value(y);
}

static cell safe_eq_p(cell x, cell y) {
	return compare("equal", real_equal_p, chr_equal_p, str_equal_p, y, x)?
			One: Zero;
}

static void op_eq(void) {
	cell	x, y;

	TWO_ARGS("equal")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("equal", safe_eq_p, y, x);
}

static cell safe_gt_p(cell x, cell y) {
	return compare("more", real_less_p, chr_less_p, str_less_p, x, y)?
			One: Zero;
}

static void op_gt(void) {
	cell	x, y;

	TWO_ARGS("more")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("more", safe_gt_p, y, x);
}

static cell safe_lt_p(cell x, cell y) {
	return compare("less", real_less_p, chr_less_p, str_less_p, y, x)?
			One: Zero;
}

static void op_lt(void) {
	cell	x, y;

	TWO_ARGS("less")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("less", safe_lt_p, y, x);
}

static void op_expand(void) {
	cell	x;

	ONE_ARG("expand");
	x = car(Dstack);
	if (integer_p(x) || list_p(x)) {
		x = expand(x);
		car(Dstack) = x;
		return;
	}
	unknown1("expand");
}

static void op_find(void) {
	cell	x, y, n;

	TWO_ARGS("find")
	y = cadr(Dstack);
	x = car(Dstack);
	if (list_p(y)) {
		n = find(x, y);
		Dstack = cdr(Dstack);
		car(Dstack) = n;
		return;
	}
	if (string_p(y) && char_p(x)) {
		n = find_string(x, y);
		Dstack = cdr(Dstack);
		car(Dstack) = n;
		return;
	}
	if (string_p(y) && string_p(x)) {
		n = find_substring(x, y);
		Dstack = cdr(Dstack);
		car(Dstack) = n;
		return;
	}
	if (dictionary_p(y) && !dictionary_p(x)) {
		x = dict_lookup(y, x);
		Dstack = cdr(Dstack);
		car(Dstack) = UNDEFINED==x? x: cadr(x);
		return;
	}
	unknown2("find");
}

static cell safe_floor(cell x) {
	if (integer_p(x)) return x;
	if (number_p(x)) {
		x = real_floor(x);
		if (real_exponent(x) < S9_MANTISSA_SIZE)
			x = real_to_bignum(x);
		return x;
	}
	return error("floor: type error", cons(x, NIL));
}

static void op_floor(void) {
	cell	x;

	ONE_ARG("floor");
	x = car(Dstack);
	monadrec("floor", safe_floor, x);
}

static int intpart(cell x) {
	x = real_floor(x);
	save(x);
	x = real_to_bignum(x);
	unsave(1);
	return intvalue(x);
}

static int fracpart(cell x) {
	cell	n;

	n = real_floor(x);
	save(n);
	n = real_subtract(x, n);
	n = real_mantissa(n);
	unsave(1);
	return intvalue(n);
}

static cell form(cell x, cell proto) {
	#define L	1024
	char	*s, *p, buf[L];
	cell	n;

	if (string_len(x) > L)
		return UNDEFINED;
	strcpy(buf, string(x));
	if (integer_p(proto)) {
		if (!integer_string_p(buf))
			return UNDEFINED;
		return string_to_bignum(buf);
	}
	if (real_p(proto)) {
		if (!string_numeric_p(buf))
			return UNDEFINED;
		return string_to_real(buf);
	}
	if (string_p(proto)) {
		return x;
	}
	if (char_p(proto)) {
		if (string_len(x) != 2)
			return UNDEFINED;
		return make_char(string(x)[0]);
	}
	if (symbol_p(proto)) {
		s = string(x);
		if (':' == s[0]) s++;
		if (!isalpha(s[0]))
			return UNDEFINED;
		for (p = s; *p; p++)
			if (!is_symbolic(*p))
				return UNDEFINED;
		if (s != string(x)) {
			n = make_string("", string_len(x)-2);
			strcpy(string(n), string(x)+1);
			save(n);
			n = string_to_symbol(n);
			unsave(1);
			return n;
		}
		return string_to_symbol(x);
	}
	return x;
}

static cell safe_form(cell x, cell y) {
	if (string_p(x)) {
		return form(x, y);
	}
	return error("form: expected string, got", x);
}

static void op_form(void) {
	cell	x, y;

	TWO_ARGS("form");
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("form", safe_form, x, y);
}

static cell format_symbol(cell x) {
	cell	n;

	n = make_string("", symbol_len(x));
	string(n)[0] = ':';
	strcpy(&string(n)[1], symbol_name(x));
	return n;
}

static cell safe_format(cell x) {
	char	b[2];

	if (string_p(x))
		return x;
	if (symbol_p(x))
		return format_symbol(x);
	if (variable_p(x))
		return format_symbol(var_symbol(x));
	if (integer_p(x))
		return bignum_to_string(x);
	if (char_p(x)) {
		b[0] = char_value(x);
		b[1] = 0;
		return make_string(b, 1);
	}
	if (real_p(x))
		return real_to_string(x, 0);
	return UNDEFINED;
}

static void op_format(void) {
	cell	x;

	ONE_ARG("format");
	x = car(Dstack);
	monadrec("format", safe_format, x);
}

static cell safe_format2(cell x, cell y) {
	cell	n, m;
	int	k, kf, kp, off, p;

	if (integer_p(y)) {
		k = intvalue(y);
		kf = 0;
	}
	else if (real_p(y)) {
		n = real_abs(y);
		k = intpart(n);
		kf = fracpart(n);
	}
	else {
		return error("format2: type error", cons(x, cons(y, NIL)));
	}
	n = safe_format(x);
	save(n);
	if (real_p(x) && strchr(string(n), 'e') == NULL && kf > 0) {
		k = abs(k);
		kp = k;
		k = abs(k) + kf+1;
		p = strlen(strchr(string(n), '.')) - 1;
		off = kp - string_len(n) + p + 2;
		if (k >= string_len(n) && p <= kf) {
			m = make_string("", k);
			memset(string(m), ' ', k);
			string(m)[k] = 0;
			memcpy(string(m)+off, string(n), string_len(n)-1);
			memset(string(m)+k-kf+p, '0', kf-p);
			n = m;
		}
	}
	else {
		if (abs(k) >= string_len(n)) {
			m = make_string("", abs(k));
			memset(string(m), ' ', abs(k));
			string(m)[abs(k)] = 0;
			if (k > 0) {
				memcpy(string(m), string(n), string_len(n)-1);
			}
			else {
				k = -k;
				memcpy(string(m)+k-string_len(n)+1, string(n),
					string_len(n));
			}
			n = m;
		}
	}
	unsave(1);
	return n;
}

static void op_format2(void) {
	cell	x, y;

	TWO_ARGS("format2");
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("format2", safe_format2, x, y);
}

static void fun(int immed, int arity) {
	cell	x;

	ONE_ARG("function");
	x = car(Dstack);
	if (list_p(x)) {
		x = make_function(x, immed, arity);
		car(Dstack) = x;
		return;
	}
	unknown1("function");
}

static void op_fun0(void) { fun(0, 0); }
static void op_fun1(void) { fun(0, 1); }
static void op_fun2(void) { fun(0, 2); }
static void op_fun3(void) { fun(0, 3); }
static void op_imm1(void) { fun(1, 1); }
static void op_imm2(void) { fun(1, 2); }

static void op_group(void) {
	cell	x;

	ONE_ARG("group");
	x = car(Dstack);
	if (list_p(x)) {
		x = group(x);
		car(Dstack) = x;
		return;
	}
	if (string_p(x)) {
		x = group_string(x);
		car(Dstack) = x;
		return;
	}
	unknown1("group");
}

static void op_first(void) {
	cell	x;

	ONE_ARG("first");
	x = car(Dstack);
	if (pair_p(x)) {
		car(Dstack) = caar(Dstack);
		return;
	}
	if (string_p(x)) {
		if (!empty_p(x))
			x = make_char(string(x)[0]);
		car(Dstack) = x;
		return;
	}
	/* identity */
}

static void op_if(void) {
	cell	x, y, z;

	THREE_ARGS("if")
	z = caddr(Dstack);
	y = cadr(Dstack);
	x = car(Dstack);
	if (list_p(y) && list_p(x)) {
		Dstack = cddr(Dstack);
		if (false_p(z))
			car(Dstack) = x;
		else
			car(Dstack) = y;
		State = S_APPIF;
		return;
	}
	unknown3("if");
}

static void op_index(void) {
	cell	x, y;

	TWO_ARGS("index")
	y = cadr(Dstack);
	x = car(Dstack);
	if (function_p(y)) {
		op_apply();
		return;
	}
	if ((list_p(y) || string_p(y)) && (list_p(x) || integer_p(x))) {
		if (string_p(y))
			y = string_to_vector(y);
		else
			y = list_to_vector(y);
		save(y);
		x = ndx(y, vector_len(y), x);
		unsave(1);
		y = cadr(Dstack);
		car(Dstack) = x; /* protect x */
		if (string_p(y))
			x = list_to_string(x);
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	unknown2("index");
}

static void op_indexd(void) {
	cell	x, y;

	TWO_ARGS("index-in-depth")
	y = cadr(Dstack);
	x = car(Dstack);
	if (function_p(y)) {
		op_apply();
		return;
	}
	if (pair_p(y) && (pair_p(x) || integer_p(x))) {
		x = ndx_in_depth(y, x);
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	unknown2("index-in-depth");
}

static cell safe_intdiv(cell x, cell y) {
	if (integer_p(x) && integer_p(y)) {
		x = bignum_divide(x, y);
		if (undefined_p(x))
			return error("division by zero", VOID);
		return car(x);
	}
	return error("integer-divide: type error", cons(x, cons(y, NIL)));
}

static void op_intdiv(void) {
	cell	x, y;

	TWO_ARGS("integer-divide")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("integer-divide", safe_intdiv, y, x);
}

static void iter(int s) {
	cell	x, y, z, n;

	THREE_ARGS("iterate")
	z = caddr(Dstack);
	y = cadr(Dstack);
	x = car(Dstack);
	if (integer_p(z) && function_p(x)) {
		if (bignum_zero_p(z)) {
			Dstack = cddr(Dstack);
			car(Dstack) = y;
		}
		else {
			n = bignum_abs(z);
			caddr(Dstack) = n;
			if (S_S_ITER == s)
				push(cons(y, NIL));
			push(Barrier);
			n = cons(x, NIL);
			push(cons(y, n));
			State = s;
		}
		return;
	}
	unknown3("iterate");
}

static void op_iter(void) {
	iter(S_ITER);
}

static void op_join(void) {
	cell	y, x, n;

	TWO_ARGS("join")
	y = cadr(Dstack);
	x = car(Dstack);
	if (dictionary_p(x) && tuple_p(y)) {
		x = dict_add(x, car(y), cadr(y));
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	if (tuple_p(x) && dictionary_p(y)) {
		x = dict_add(y, car(x), cadr(x));
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	n = join(y, x);
	Dstack = cdr(Dstack);
	car(Dstack) = n;
}

static void op_list(void) {
	cell	x, n;

	ONE_ARG("list");
	x = car(Dstack);
	if (char_p(x)) {
		n = make_string("", 1);
		string(n)[0] = char_value(x);
		string(n)[1] = 0;
		car(Dstack) = n;
		return;
	}
	x = cons(x, NIL);
	car(Dstack) = x;
}

static void op_match(void) {
	cell	x, y;

	TWO_ARGS("match")
	y = cadr(Dstack);
	x = car(Dstack);
	x = match(x, y)? One: Zero;
	Dstack = cdr(Dstack);
	car(Dstack) = x;
}

static cell safe_max(cell x, cell y) {
	if (number_p(y) && number_p(x)) {
		return real_less_p(x, y)? y: x;
	}
	return error("max: type error", cons(x, cons(y, NIL)));
}

static void op_max(void) {
	cell	x, y;

	TWO_ARGS("max")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("max", safe_max, y, x);
}

static cell safe_min(cell x, cell y) {
	if (number_p(y) && number_p(x)) {
		return real_less_p(x, y)? x: y;
	}
	return error("min: type error", cons(x, cons(y, NIL)));
}

static void op_min(void) {
	cell	x, y;

	TWO_ARGS("min")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("min", safe_min, y, x);
}

static cell safe_real_subtract(cell x, cell y) {
	if (number_p(x) && number_p(y))
		return real_subtract(x, y);
	return error("minus: type error", cons(x, cons(y, NIL)));
}

static void op_minus(void) {
	cell	x, y;

	TWO_ARGS("minus")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("minus", safe_real_subtract, y, x);
}

static cell safe_negate(cell x) {
	if (number_p(x))
		return real_negate(x);
	return error("negate: type error", cons(x, NIL));
}

static void op_neg(void) {
	cell	x;

	ONE_ARG("negate");
	x = car(Dstack);
	monadrec("negate", safe_negate, x);
}

static void op_newdict(void) {
	cell	x;

	x = copy_dict(car(Dstack));
	car(Dstack) = x;
}

static cell safe_not(cell x) {
	return false_p(x)? One: Zero;
}

static void op_not(void) {
	cell	x;

	ONE_ARG("not");
	x = car(Dstack);
	monadrec("negate", safe_not, x);
}

static void over(int s) {
	cell	x, y, n;

	TWO_ARGS("over")
	y = cadr(Dstack);
	x = car(Dstack);
	if ((list_p(y) || string_p(y)) && function_p(x)) {
		n = cons(NIL, cddr(Dstack));
		cddr(Dstack) = n;
		if (NIL == y || empty_p(y)) {
			Dstack = cddr(Dstack);
			car(Dstack) = y;
		}
		else if (NIL == cdr(y)) {
			Dstack = cddr(Dstack);
			if (S_OVER == s)
				y = car(y);
			car(Dstack) = y;
		}
		else if (string_p(y) && string_len(y) < 3) {
			Dstack = cddr(Dstack);
			y = make_char(string(y)[0]);
			if (S_S_OVER == s)
				y = cons(y, NIL);
			car(Dstack) = y;
		}
		else {
			if (string_p(y)) {
				y = string_to_list(y, STRING_NIL);
			}
			cadr(Dstack) = y;
			push(cons(car(y), NIL));
			push(Barrier);
			n = cons(x, NIL);
			n = cons(cadr(y), n);
			push(cons(car(y), n));
			cadr(cdddr(Dstack)) = cddr(y);
			State = s;
		}
		return;
	}
	if (S_S_OVER == s)
		y = cons(y, NIL);
	Dstack = cdr(Dstack);
	car(Dstack) = y;
}

static void op_over(void) {
	over(S_OVER);
}

static void over_n(int s) {
	cell	x, y, z, n;

	THREE_ARGS("over/n")
	z = caddr(Dstack);
	y = cadr(Dstack);
	x = car(Dstack);
	if ((list_p(y) || string_p(y)) && function_p(x)) {
		if (NIL == y || empty_p(y)) {
			Dstack = cddr(Dstack);
			car(Dstack) = z;
		}
		else {
			if (string_p(y)) {
				y = string_to_list(y, STRING_NIL);
			}
			save(y);
			cadr(Dstack) = cdr(y);
			push(cons(z, NIL));
			push(Barrier);
			n = cons(x, NIL);
			n = cons(car(y), n);
			push(cons(z, n));
			State = s;
			unsave(1);
		}
		return;
	}
	if (function_p(x)) {
		n = NIL;
		if (S_S_OVER == s) {
			n = cons(S_join, n);
			n = cons(S_list, n);
		}
		n = cons(x, n);
		n = cons(y, n);
		n = cons(z, n);
		if (S_S_OVER == s) n = cons(z, n);
		Dstack = cddr(Dstack);
		car(Dstack) = n;
		State = S_APPLY;
		return;
	}
	unknown3("over/n");
}

static void op_over_n(void) {
	over_n(S_OVER);
}

static cell safe_real_add(cell x, cell y) {
	if (number_p(x) && number_p(y))
		return real_add(x, y);
	return error("plus: type error", cons(x, cons(y, NIL)));
}

static void op_plus(void) {
	cell	x, y;

	TWO_ARGS("plus")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("plus", safe_real_add, y, x);
}

static void op_pop0(void) {
	unsave_vars();
}

static void op_pop1(void) {
	cdr(Dstack) = cddr(Dstack);
	unsave_vars();
}

static void op_pop2(void) {
	cdr(Dstack) = cdddr(Dstack);
	unsave_vars();
}

static void op_pop3(void) {
	cdr(Dstack) = cddddr(Dstack);
	unsave_vars();
}

static cell safe_power(cell x, cell y) {
	if (number_p(x) && number_p(y)) {
		return real_power(x, y);
	}
	return error("power: type error", cons(x, cons(y, NIL)));
}

static void op_pow(void) {
	cell	x, y;

	TWO_ARGS("power")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("power", safe_power, y, x);
}

static void op_range(void) {
	cell	x;

	ONE_ARG("range");
	x = car(Dstack);
	if (list_p(x) || string_p(x)) {
		x = range(x);
		car(Dstack) = x;
		return;
	}
	unknown1("range");
}

static cell safe_recip(cell x) {
	if (number_p(x))
		return real_divide(One, x);
	return error("reciprocal: type error", cons(x, NIL));
}

static void op_recip(void) {
	cell	x;

	ONE_ARG("reciprocal");
	x = car(Dstack);
	monadrec("reciprocal", safe_recip, x);
}

static cell safe_rem(cell x, cell y) {
	if (integer_p(y) && integer_p(x)) {
		x = bignum_divide(x, y);
		if (undefined_p(x))
			return error("division by zero", VOID);
		return cdr(x);
	}
	return error("rem: type error", cons(x, cons(y, NIL)));
}

static void op_rem(void) {
	cell	x, y;

	TWO_ARGS("rem")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("rem", safe_rem, y, x);
}

static void op_reshape(void) {
	cell	y, x;

	TWO_ARGS("reshape");
	y = cadr(Dstack);
	x = car(Dstack);
	if (integer_p(y)) {
		if (bignum_zero_p(y)) {
			pop();
			car(Dstack) = x;
			return;
		}
		binop(reshape, y, x);
		return;
	}
	if (list_p(y)) {
		binop(reshape, y, x);
		return;
	}
	unknown2("reshape");
}

static void op_rot(void) {
	cell	x, y;

	TWO_ARGS("rotate");
	y = cadr(Dstack);
	x = car(Dstack);
	if (atom_p(x) && !string_p(x)) {
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	if (integer_p(y) && list_p(x)) {
		x = rotate(x, y);
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	if (integer_p(y) && string_p(x)) {
		x = rotate_string(x, y);
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	unknown2("rotate");
}

static void op_rev(void) {
	cell	x;

	ONE_ARG("reverse");
	x = car(Dstack);
	if (list_p(x)) {
		x = rev(car(Dstack));
		car(Dstack) = x;
		return;
	}
	if (string_p(x)) {
		x = rev_string(car(Dstack));
		car(Dstack) = x;
		return;
	}
	/* identity */
}

static void op_s_conv(void) {
	conv(S_S_CONV);
}

static void op_s_iter(void) {
	iter(S_S_ITER);
}

static void op_s_over(void) {
	over(S_S_OVER);
}

static void op_s_over_n(void) {
	over_n(S_S_OVER);
}

static void op_shape(void) {
	cell	x;

	ONE_ARG("shape");
	x = car(Dstack);
	if (string_p(x) && string_len(x) > 1) {
		x = cons(make_integer(string_len(x)-1), NIL);
		car(Dstack) = x;
		return;
	}
	if (pair_p(x)) {
		x = shape(car(Dstack));
		car(Dstack) = x;
		return;
	}
	car(Dstack) = Zero;
}

static void op_size(void) {
	cell	x, new;

	ONE_ARG("size");
	x = car(Dstack);
	if (list_p(x)) {
		x = make_integer(length(x));
		car(Dstack) = x;
		return;
	}
	if (number_p(x)) {
		x = real_abs(x);
		car(Dstack) = x;
		return;
	}
	if (char_p(x)) {
		new = make_integer(char_value(x));
		car(Dstack) = new;
		return;
	}
	if (string_p(x)) {
		x = make_integer(string_len(x)-1);
		car(Dstack) = x;
		return;
	}
	if (dictionary_p(x)) {
		x = make_integer(dict_size(x));
		car(Dstack) = x;
		return;
	}
	unknown1("size");
}

static void op_split(void) {
	cell	x, y;

	TWO_ARGS("split")
	y = cadr(Dstack);
	x = car(Dstack);
	if ((pair_p(y) || integer_p(y)) && list_p(x)) {
		binop(split, y, x);
		return;
	}
	if ((pair_p(y) || integer_p(y)) && string_p(x)) {
		binop(split_string, y, x);
		return;
	}
	unknown2("split");
}

static void op_swap(void) {
	cell	x;

	TWO_ARGS("swap")
	x = car(Dstack);
	car(Dstack) = cadr(Dstack);
	cadr(Dstack) = x;
}

static void sysfn(int id);

static void op_syscall(void) {
	cell	x;
	int	id;

	ONE_ARG("syscall");
	x = car(Dstack);
	if (integer_p(x)) {
		id = intvalue(x);
		Dstack = cdr(Dstack);
		sysfn(id);
		return;
	}
	unknown1("syscall");
}

static void op_take(void) {
	cell	x, y;

	TWO_ARGS("take")
	y = cadr(Dstack);
	x = car(Dstack);
	if (integer_p(y) && list_p(x)) {
		x = take(x, intvalue(y));
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	if (integer_p(y) && string_p(x)) {
		x = take_string(x, intvalue(y));
		Dstack = cdr(Dstack);
		car(Dstack) = x;
		return;
	}
	unknown2("take");
}

static cell safe_real_multiply(cell x, cell y) {
	if (number_p(x) && number_p(y))
		return real_multiply(x, y);
	return error("times: type error", cons(x, cons(y, NIL)));
}

static void op_times(void) {
	cell	x, y;

	TWO_ARGS("times")
	y = cadr(Dstack);
	x = car(Dstack);
	dyadrec("times", safe_real_multiply, y, x);
}

static void op_transp(void) {
	cell	x;

	ONE_ARG("transpose");
	x = car(Dstack);
	if (list_p(x)) {
		x = transpose(x);
		car(Dstack) = x;
		return;
	}
	unknown1("transpose");
}

static void op_up(void) {
	cell	x;

	ONE_ARG("grade-up");
	x = car(Dstack);
	if (list_p(x)) {
		x = grade(less, x);
		car(Dstack) = x;
		return;
	}
	if (string_p(x)) {
		x = string_to_list(x, NIL);
		car(Dstack) = x;
		x = grade(less, x);
		car(Dstack) = x;
		return;
	}
	unknown1("grade-up");
}

static void op_undef(void) {
	cell	x;

	ONE_ARG("undefined");
	x = car(Dstack);
	car(Dstack) = UNDEFINED == x? One: Zero;
}

static void op_while(void) {
	cell	x, y, z, n;

	THREE_ARGS("while")
	z = caddr(Dstack);
	y = cadr(Dstack);
	x = car(Dstack);
	if (function_p(z) && function_p(x)) {
		push(Barrier);
		n = cons(S_pop1, NIL);
		n = cons(S_call1, n);
		n = cons(z, n);
		push(cons(y, n));
		State = S_WPRED;
		return;
	}
	unknown3("while");
}

static void op_s_while(void) {
	cell	x, y, z, n;

	THREE_ARGS("while")
	z = caddr(Dstack);
	y = cadr(Dstack);
	x = car(Dstack);
	if (function_p(z) && function_p(x)) {
		n = cons(y, NIL);
		cadr(Dstack) = n;
		push(Barrier);
		n = cons(S_pop1, NIL);
		n = cons(S_call1, n);
		n = cons(z, n);
		push(cons(y, n));
		State = S_S_WPRED;
		return;
	}
	unknown3("while");
}

static void op_x(void) {
	push(NIL == Frame? UNDEFINED: car(Frame));
}

static void op_y(void) {
	push(NIL == Frame || NIL == cdr(Frame)? UNDEFINED: cadr(Frame));
}

static void op_z(void) {
	push(NIL == Frame || NIL == cdr(Frame) || NIL == cddr(Frame)?
		UNDEFINED: caddr(Frame));
}

OP Ops[] = {
	{ "%amend",   0, op_amend   },
	{ "%amendd",  0, op_amendd  },
	{ "%atom",    0, op_atom    },
	{ "%call0",   1, op_call0   },
	{ "%call1",   1, op_call1   },
	{ "%call2",   1, op_call2   },
	{ "%call3",   1, op_call3   },
	{ "%char",    0, op_char    },
	{ "%clear",   0, op_clear   },
	{ "%conv",    1, op_conv    },
	{ "%cut",     0, op_cut     },
	{ "%def",     0, op_def     },
	{ "%div",     0, op_div     },
	{ "%down",    0, op_down    },
	{ "%drop",    0, op_drop    },
	{ "%each",    1, op_each    },
	{ "%each2",   1, op_each2   },
	{ "%eachl",   1, op_eachl   },
	{ "%eachp",   1, op_eachp   },
	{ "%eachr",   1, op_eachr   },
	{ "%enum",    0, op_enum    },
	{ "%eq",      0, op_eq      },
	{ "%expand",  0, op_expand  },
	{ "%find",    0, op_find    },
	{ "%first",   0, op_first   },
	{ "%floor",   0, op_floor   },
	{ "%form",    0, op_form    },
	{ "%format",  0, op_format  },
	{ "%format2", 0, op_format2 },
	{ "%fun0",    0, op_fun0    },
	{ "%fun1",    0, op_fun1    },
	{ "%fun2",    0, op_fun2    },
	{ "%fun3",    0, op_fun3    },
	{ "%group",   0, op_group   },
	{ "%gt",      0, op_gt      },
	{ "%if",      1, op_if      },
	{ "%imm1",    0, op_imm1    },
	{ "%imm2",    0, op_imm2    },
	{ "%index",   1, op_index   },
	{ "%indexd",  1, op_indexd  },
	{ "%intdiv",  0, op_intdiv  },
	{ "%iter",    1, op_iter    },
	{ "%join",    0, op_join    },
	{ "%list",    0, op_list    },
	{ "%lt",      0, op_lt      },
	{ "%match",   0, op_match   },
	{ "%max",     0, op_max     },
	{ "%min",     0, op_min     },
	{ "%minus",   0, op_minus   },
	{ "%neg",     0, op_neg     },
	{ "%newdict", 0, op_newdict },
	{ "%not",     0, op_not     },
	{ "%over",    1, op_over    },
	{ "%over2",   1, op_over_n  },
	{ "%plus",    0, op_plus    },
	{ "%pop0",    0, op_pop0    },
	{ "%pop1",    0, op_pop1    },
	{ "%pop2",    0, op_pop2    },
	{ "%pop3",    0, op_pop3    },
	{ "%power",   0, op_pow     },
	{ "%range",   0, op_range   },
	{ "%recip",   0, op_recip   },
	{ "%rem",     0, op_rem     },
	{ "%reshape", 0, op_reshape },
	{ "%rev",     0, op_rev     },
	{ "%rot",     0, op_rot     },
	{ "%sconv",   1, op_s_conv  },
	{ "%siter",   1, op_s_iter  },
	{ "%sover",   1, op_s_over  },
	{ "%sover2",  1, op_s_over_n},
	{ "%swhile",  1, op_s_while },
	{ "%shape",   0, op_shape   },
	{ "%size",    0, op_size    },
	{ "%siter",   0, op_s_iter  },
	{ "%split",   0, op_split   },
	{ "%swap",    0, op_swap    },
	{ "%syscall", 0, op_syscall },
	{ "%take",    0, op_take    },
	{ "%times",   0, op_times   },
	{ "%transp",  0, op_transp  },
	{ "%up",      0, op_up      },
	{ "%undef",   0, op_undef   },
	{ "%while",   1, op_while   },
	{ "%x",       0, op_x       },
	{ "%y",       0, op_y       },
	{ "%z",       0, op_z       },
	{ NULL }
};

/*
 * Built-in System Functions
 */

static void sys_close(void) {
	cell	x;

	ONE_ARG(".cc");
	x = car(Dstack);
	push(car(Dstack));
	if (input_port_p(x) || output_port_p(x)) {
		if (x == From_chan) {
			set_input_port(0);
			From_chan = 0;
		}
		if (x == To_chan) {
			set_output_port(1);
			To_chan = 1;
		}
		if (	port_no(x) != 0 &&
			port_no(x) != 1 &&
			port_no(x) != 2)
		{
			close_port(port_no(x));
		}
		car(Dstack) = NIL;
		return;
	}
	unknown1(".cc");
}

static void sys_comment(void) {
	cell	x;
	char	msg[80];
	int	sln, k;

	ONE_ARG(".comment");
	x = car(Dstack);
	push(car(Dstack));
	if (string_p(x)) {
		sln = Line;
		k = strlen(string(x));
		set_input_port(Prog_chan);
		for (;;) {
			if (kg_getline(Inbuf, TOKEN_LENGTH) == NULL) {
				sprintf(msg, "undelimited comment (line %d)", 
					sln);
				error(msg, VOID);
				break;
			}
			if (!strncmp(string(x), Inbuf, k))
				break;
		}
		return;
	}
	unknown1(".comment");
}

static void sys_delete(void) {
	cell	x;

	ONE_ARG(".df");
	x = car(Dstack);
	push(car(Dstack));
	if (string_p(x)) {
		if (remove(string(x)) < 0)
			error("could not delete", x);
		return;
	}
	unknown1(".df");
}

static void sys_display(void) {
	cell	x;

	ONE_ARG(".p");
	x = car(Dstack);
	Display = 1;
	if (!outport_open_p())
		error(".d: writing to closed channel",
			make_port(To_chan, T_OUTPUT_PORT));
	else
		kg_write(x);
	Display = 0;
	push(car(Dstack));
}

static cell evalstr(cell x);

static void sys_eval(void) {
	cell	x;

	ONE_ARG(".E");
	x = car(Dstack);
	if (string_p(x)) {
		x = evalstr(x);
		car(Dstack) = x;
		push(car(Dstack));
		return;
	}
	unknown1(".E");
}

static void sys_flush(void) {
	if (!outport_open_p())
		error(".fl: writing to closed channel",
			make_port(To_chan, T_OUTPUT_PORT));
	else
		flush();
	push(NIL);
}

static void sys_fromchan(void) {
	cell	x, old;

	ONE_ARG(".fc");
	x = car(Dstack);
	push(car(Dstack));
	old = From_chan;
	if (false_p(x)) {
		set_input_port(0);
		old = make_port(old, T_INPUT_PORT);
		car(Dstack) = old;
		return;
	}
	if (input_port_p(x)) {
		From_chan = port_no(x);
		set_input_port(port_no(x));
		old = make_port(old, T_INPUT_PORT);
		car(Dstack) = old;
		return;
	}
	unknown1(".fc");
}

static void sys_infile(void) {
	cell	x;
	int	p;

	ONE_ARG(".ic");
	x = car(Dstack);
	push(car(Dstack));
	if (string_p(x)) {
		if ((p = open_input_port(string(x))) < 0) {
			error(".ic: failed to open input file", x);
			return;
		}
		x = make_port(p, T_INPUT_PORT);
		car(Dstack) = x;
		return;
	}
	unknown1(".ic");
}

static void sys_load(void) {
	cell	x;

	ONE_ARG(".l");
	x = car(Dstack);
	push(car(Dstack));
	if (string_p(x)) {
		x = load(x, 0, 0);
		car(Dstack) = x;
		return;
	}
	unknown1(".l");
}

static void sys_module(void) {
	cell	x;

	ONE_ARG(".module");
	x = car(Dstack);
	push(car(Dstack));
	if (symbol_p(x)) {
		if (Module != UNDEFINED) {
			error("nested module; contained in",
				make_symbol(Modname, strlen(Modname)));
		}
		else {
			Module = NIL;
			Mod_funvars = NIL;
			strcpy(Modname, symbol_name(x));
		}
		return;
	}
	if (false_p(x)) {
		if (UNDEFINED == Module)
			error("no module open", VOID);
		Module = UNDEFINED;
		return;
	}
	unknown1(".module");
}

static void sys_more(void) {
	ONE_ARG(".mi");
	if (!inport_open_p())
		error(".mi: testing closed channel",
			make_port(From_chan, T_INPUT_PORT));
	push(port_eof(input_port())? Zero: One);
}

static void outfile(int append) {
	cell	x;
	int	p;

	ONE_ARG(".oc");
	x = car(Dstack);
	push(car(Dstack));
	if (string_p(x)) {
		if ((p = open_output_port(string(x), append)) < 0) {
			if (append)
				error(".ac: failed to open output file", x);
			else
				error(".oc: failed to open output file", x);
			return;
		}
		x = make_port(p, T_OUTPUT_PORT);
		car(Dstack) = x;
		return;
	}
	unknown1(".oc");
}

static void sys_outfile(void) {
	outfile(0);
}

static void sys_appfile(void) {
	outfile(1);
}

static void sys_print(void) {
	if (!outport_open_p())
		error(".p: writing to closed channel",
			make_port(To_chan, T_OUTPUT_PORT));
	else
		sys_display();
	nl();
}

static void sys_randnum(void) {
	cell	x, n, e = 0;

	n = rand() % S9_INT_SEG_LIMIT;
	x = 0;
	while (n > 0) {
		x = x*10+n%10;
		e++;
		n /= 10;
	}
	x = make_real(1, -e, make_integer(x));
	push(x);
}

static void sys_read(void) {
	if (!inport_open_p())
		error(".r: reading from closed channel",
			make_port(From_chan, T_INPUT_PORT));
	else {
		push(kg_read());
		while (readc() != '\n' && readc() != EOF)
			;
	}
}

static void sys_readln(void) {
	cell	x = NIL;
	int	c;

	save(x);
	if (!inport_open_p()) {
		error(".rl: reading from closed channel",
			make_port(From_chan, T_INPUT_PORT));
	}
	else {
		while ((c = readc()) != '\n') {
			if (EOF == c)
				break;
			x = cons(make_char(c), x);
			car(Stack) = x;
		}
		x = rev(x);
	}
	car(Stack) = x;
	x = list_to_string(x);
	unsave(1);
	push(x);
}

static cell compile(char *p);

static void sys_readstr(void) {
	cell	x;

	if (s9_aborted())
		return;
	ONE_ARG(".rs");
	x = car(Dstack);
	if (string_p(x)) {
		Report = 0;
		x = compile(string(x));
		Report = 1;
		car(Dstack) = car(x);
		if (s9_aborted()) {
			s9_reset();
			car(Dstack) = UNDEFINED;
		}
		push(car(Dstack));
		return;
	}
	unknown1(".rs");
}

static void sys_system(void) {
	cell	x;
	int	r;

#ifdef SAFE
	error("shell access disabled", VOID);
	return;
#endif
	ONE_ARG(".sys");
	x = car(Dstack);
	if (string_p(x)) {
		r = system(string(x));
		x = make_integer(r);
		car(Dstack) = x;
		push(car(Dstack));
		return;
	}
	unknown1(".sys");
}

#ifndef plan9
static void sys_pclock(void) {
	clock_t	t;
	cell	x, y;

	t = clock();
	x = make_integer(t);
	save(x);
	y = make_integer(10000000);
	x = bignum_multiply(x, y);
	car(Stack) = x;
	y = make_integer(CLOCKS_PER_SEC);
	x = bignum_divide(x, y);
	x = car(x);
	x = make_real(1, -7, x);
	unsave(1);
	push(x);
}
#endif

static void sys_tochan(void) {
	cell	x, old;

	ONE_ARG(".tc");
	x = car(Dstack);
	push(car(Dstack));
	old = To_chan;
	if (false_p(x)) {
		set_output_port(1);
		old = make_port(old, T_OUTPUT_PORT);
		car(Dstack) = old;
		return;
	}
	if (output_port_p(x)) {
		To_chan = port_no(x);
		set_output_port(port_no(x));
		old = make_port(old, T_OUTPUT_PORT);
		car(Dstack) = old;
		return;
	}
	unknown1(".tc");
}

static void sys_write(void) {
	ONE_ARG(".w");
	if (!outport_open_p())
		error(".w: writing to closed channel",
			make_port(To_chan, T_OUTPUT_PORT));
	else
		kg_write(car(Dstack));
	push(car(Dstack));
}

static void sys_exit(void) {
	cell	x;

	ONE_ARG(".x");
	x = car(Dstack);
	bye(false_p(x) == 0);
}

SYS Sysfns[] = {
	{ ".ac",      1, sys_appfile },
	{ ".cc",      1, sys_close   },
	{ ".comment", 1, sys_comment },
	{ ".df",      1, sys_delete  },
	{ ".d",       1, sys_display },
	{ ".E",       1, sys_eval    },
	{ ".fc",      1, sys_fromchan},
	{ ".fl",      0, sys_flush   },
	{ ".ic",      1, sys_infile  },
	{ ".l",       1, sys_load    },
	{ ".mi",      1, sys_more    },
	{ ".module",  1, sys_module  },
	{ ".oc",      1, sys_outfile },
	{ ".p",       1, sys_print   },
#ifndef plan9
	{ ".pc",      0, sys_pclock  },
#endif
	{ ".r",       0, sys_read    },
	{ ".rl",      0, sys_readln  },
	{ ".rn",      0, sys_randnum },
	{ ".rs",      1, sys_readstr },
	{ ".sys",     1, sys_system  },
	{ ".tc",      1, sys_tochan  },
	{ ".w",       1, sys_write   },
	{ ".x",       1, sys_exit    },
	{ NULL }
};

static void sysfn(int id) {
	(*Sysfns[id].handler)();
}

/*
 * Virtual Machine, Adverb Handlers
 */

static cell next_conv(cell s) {
	cell	y0, y1;
	cell	n;

	y0 = cadr(Dstack);
	y1 = car(Dstack);
	if (match(y0, y1)) {
		n = y0;
		Dstack = cdddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	cadr(Dstack) = y1;
	car(Dstack) = Barrier;
	n = cons(caddr(Dstack), NIL);
	push(cons(y1, n));
	return s;
}

static cell next_each(cell s) {
	cell	x = cdr(Dstack), y = cddr(Dstack);
	cell	n;

	if (NIL == car(y) || STRING_NIL == car(y)) {
		n = revb(car(Dstack));
		car(Dstack) = n; /* protect x */
		if (STRING_NIL == car(y))
			n = list_to_string(n);
		Dstack = cddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	push(Barrier);
	n = cons(car(x), NIL);
	push(cons(caar(y), n));
	car(y) = cdar(y);
	return s;
}

static cell next_each2(cell s) {
	cell	x = cdr(Dstack), y = cddr(Dstack), z = cdddr(Dstack);
	cell	n;

	if (	NIL == car(y) || NIL == car(z) ||
		STRING_NIL == car(y) || STRING_NIL == car(z)
	) {
		n = revb(car(Dstack));
		if (STRING_NIL == car(y) || STRING_NIL == car(z))
			n = list_to_string(n);
		Dstack = cdddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	push(Barrier);
	n = cons(car(x), NIL);
	n = cons(caar(y), n);
	push(cons(caar(z), n));
	car(y) = cdar(y);
	car(z) = cdar(z);
	return s;
}

static cell next_eachl(cell s) {
	cell	x = cdr(Dstack), y = cddr(Dstack), z = cdddr(Dstack);
	cell	n;

	if (NIL == car(y) || STRING_NIL == car(y)) {
		n = revb(car(Dstack));
		if (STRING_NIL == car(y))
			n = list_to_string(n);
		Dstack = cdddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	push(Barrier);
	n = cons(car(x), NIL);
	n = cons(caar(y), n);
	push(cons(car(z), n));
	car(y) = cdar(y);
	return s;
}

static cell next_eachp(cell s) {
	cell	x = cdr(Dstack), y = cddr(Dstack);
	cell	n;

	if (	NIL == car(y) || NIL == cdar(y) ||
		STRING_NIL == car(y) || STRING_NIL == cdar(y)
	) {
		n = revb(car(Dstack));
		if (STRING_NIL == car(y) || STRING_NIL == cdar(y))
			n = list_to_string(n);
		Dstack = cddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	push(Barrier);
	n = cons(car(x), NIL);
	n = cons(cadar(y), n);
	push(cons(caar(y), n));
	car(y) = cdar(y);
	return s;
}

static cell next_eachr(cell s) {
	cell	x = cdr(Dstack), y = cddr(Dstack), z = cdddr(Dstack);
	cell	n;

	if (NIL == car(y) || STRING_NIL == car(y)) {
		n = revb(car(Dstack));
		if (STRING_NIL == car(y))
			n = list_to_string(n);
		Dstack = cdddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	push(Barrier);
	n = cons(car(x), NIL);
	n = cons(car(z), n);
	push(cons(caar(y), n));
	car(y) = cdar(y);
	return s;
}

static cell next_over(cell s) {
	cell	x = cdr(Dstack), y = cddr(Dstack);
	cell	n;

	if (NIL == car(y) || STRING_NIL == car(y)) {
		n = car(Dstack);
		if (STRING_NIL == car(y))
			n = list_to_string(n);
		Dstack = cdddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	push(Barrier);
	n = cons(car(x), NIL);
	n = cons(caar(y), n);
	push(cons(cadr(Dstack), n));
	car(y) = cdar(y);
	return s;
}

static cell next_iter(cell s) {
	cell	x = cdr(Dstack), z = cdddr(Dstack);
	cell	n;

	n = bignum_subtract(car(z), One);
	car(z) = n;
	if (bignum_zero_p(car(z))) {
		n = car(Dstack);
		Dstack = cdddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	n = cons(car(x), NIL);
	n = cons(car(Dstack), n);
	car(Dstack) = Barrier;
	push(n);
	return s;
}

static cell next_s_conv(cell s) {
	cell	ys, y1;
	cell	n;

	ys = cadr(Dstack);
	y1 = car(Dstack);
	if (match(car(ys), y1)) {
		n = ys;
		Dstack = cdddr(Dstack);
		car(Dstack) = revb(n);
		return cdr(s);
	}
	n = cons(y1, ys);
	cadr(Dstack) = n;
	car(Dstack) = Barrier;
	n = cons(caddr(Dstack), NIL);
	push(cons(y1, n));
	return s;
}

static cell next_s_iter(cell s) {
	cell	x = cdr(Dstack), z = cdddr(Dstack);
	cell	n;

	n = bignum_subtract(car(z), One);
	car(z) = n;
	if (bignum_zero_p(car(z))) {
		n = car(Dstack);
		Dstack = cdddr(Dstack);
		car(Dstack) = revb(n);
		return cdr(s);
	}
	n = cons(car(x), NIL);
	n = cons(caar(Dstack), n);
	push(Barrier);
	push(n);
	return s;
}

static cell next_s_over(cell s) {
	cell	x = cdr(Dstack), y = cddr(Dstack);
	cell	n;

	if (NIL == car(y) || STRING_NIL == car(y)) {
		if (STRING_NIL == car(y))
			conv_to_strlst(car(Dstack));
		n = revb(car(Dstack));
		Dstack = cdddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	push(Barrier);
	n = cons(car(x), NIL);
	n = cons(caar(y), n);
	push(cons(caadr(Dstack), n));
	car(y) = cdar(y);
	return s;
}

static cell next_s_wpred(cell s) {
	cell	x = cdr(Dstack), y = cddr(Dstack);
	cell	n;

	if (false_p(car(Dstack))) {
		n = car(y);
		Dstack = cdddr(Dstack);
		if (NIL != n)
			n = cdr(n);
		car(Dstack) = revb(n);
		return cdr(s);
	}
	n = cons(car(x), NIL);
	n = cons(caar(y), n);
	car(Dstack) = Barrier;
	push(n);
	car(S) = S_S_WEXPR;
	return s;
}

static cell next_s_wexpr(cell s) {
	cell	y = cddr(Dstack), z = cdddr(Dstack);
	cell	n;

	n = cons(car(Dstack), car(y));
	car(y) = n;
	n = cons(S_pop1, NIL);
	n = cons(S_call1, n);
	n = cons(car(z), n);
	n = cons(caar(y), n);
	car(Dstack) = Barrier;
	push(n);
	car(S) = S_S_WPRED;
	return s;
}

static cell next_wpred(cell s) {
	cell	x = cdr(Dstack), y = cddr(Dstack);
	cell	n;

	if (false_p(car(Dstack))) {
		n = car(y);
		Dstack = cdddr(Dstack);
		car(Dstack) = n;
		return cdr(s);
	}
	n = cons(car(x), NIL);
	n = cons(car(y), n);
	car(Dstack) = Barrier;
	push(n);
	car(S) = S_WEXPR;
	return s;
}

static cell next_wexpr(cell s) {
	cell	y = cddr(Dstack), z = cdddr(Dstack);
	cell	n;

	car(y) = car(Dstack);
	n = cons(S_pop1, NIL);
	n = cons(S_call1, n);
	n = cons(car(z), n);
	n = cons(car(y), n);
	car(Dstack) = Barrier;
	push(n);
	car(S) = S_WPRED;
	return s;
}

static void acc(void) {
	cell	n;

	n = cons(car(Dstack), cadr(Dstack));
	Dstack = cdr(Dstack);
	car(Dstack) = n;
}

static void mov(void) {
	cell	n;

	n = car(Dstack);
	Dstack = cdr(Dstack);
	car(Dstack) = n;
}

static cell next(cell s) {
	cell	n;

	if (cadr(Dstack) != Barrier)
		return error("adverb arity error", VOID);
	n = car(Dstack);
	Dstack = cdr(Dstack);
	car(Dstack) = n;
	switch (car(s)) {
	case S_EACH:	acc(); return next_each(s);
	case S_EACH2:	acc(); return next_each2(s);
	case S_EACHL:	acc(); return next_eachl(s);
	case S_EACHP:	acc(); return next_eachp(s);
	case S_EACHR:	acc(); return next_eachr(s);
	case S_OVER:	mov(); return next_over(s);
	case S_CONV:	return next_conv(s);
	case S_ITER:	return next_iter(s);
	case S_WPRED:	return next_wpred(s);
	case S_WEXPR:	return next_wexpr(s);
	case S_S_OVER:	acc(); return next_s_over(s);
	case S_S_CONV:	return next_s_conv(s);
	case S_S_ITER:	acc(); return next_s_iter(s);
	case S_S_WPRED:	return next_s_wpred(s);
	case S_S_WEXPR:	return next_s_wexpr(s);
	default:	error("stack dump", Dstack);
			fatal("bad state");
	}
	return UNDEFINED; /*LINT*/
}

/*
 * Virtual Machine, Interpreter
 */

static cell savestate(cell f, cell fr, cell s) {
	if (NIL == f) {
		return s;
	}
	return cons(f, cons(fr, s));
}

static void cleartrace(void) {
	int	i;

	for (i=0; i<NTRACE; i++)
		Trace[i] = UNDEFINED;
	Traceptr = 0;
}

static void trace(cell v) {
	if (function_p(var_value(v))) {
		if (Traceptr >= NTRACE)
			Traceptr = 0;
		Trace[Traceptr++] = var_symbol(v);
	}
}

static void eval(cell x) {
	cell		y;
	int		skip = 0;
	extern int	Abort_flag;

	save(x);
	save(S);
	save(F);
	save(Frame);
	Frame = Dstack;
	S = NIL;
	F = cdr(x);
	x = car(x);
	State = S_EVAL;
	cleartrace();
	for (s9_reset(); 0 == Abort_flag;) {
		if (skip) {
			skip = 0;
		}
		else {
			if (variable_p(x)) {
				if (Debug) {
					trace(x);
				}
				y = var_value(x);
				if (NO_VALUE == y) {
					error("undefined", x);
					break;
				}
				x = y;
			}
			if (primop_p(x)) {
				(Ops[primop_slot(x)].handler)();
			}
			else if (syntax_p(x)) {
				State = S_EVAL;
				(Ops[primop_slot(x)].handler)();
				if (State != S_EVAL) {
					S = savestate(F, Frame, S);
					if (State != S_APPIF)
						Frame = cdr(Dstack);
					if (	S_APPLY == State ||
						S_APPIF == State
					)
						State = S_EVAL;
					else
						S = new_atom(State, S);
					F = pop();
				}
			}
			else if (function_p(x) && fun_immed(x)) {
				S = savestate(F, Frame, S);
				Frame = Dstack;
				F = fun_body(x);
			}
			else {
				push(x);
			}
		}
		if (F != NIL) {
			x = car(F);
			F = cdr(F);
		}
		else if (S != NIL) {
			if (atom_p(S)) {
				y = S;
				S = next(S);
				/*
				 * next_*() uses return cdr(s) to exit,
				 * so y =/= S is certain in this case.
				 */
				if (y == S)
					F = pop();
				skip = 1;
			}
			else {
				F = car(S);
				Frame = cadr(S);
				S = cddr(S);
				x = car(F);
				F = cdr(F);
			}
		}
		else {
			Frame = NIL;
			break;
		}
	}
	Frame = unsave(1);
	F = unsave(1);
	S = unsave(1);
	unsave(1);
}

/*
 * Bytecode Compiler, Parser
 */

static int adverb_arity(cell a, int ctx) {
	int	c0, c1;

	c0 = symbol_name(a)[0];
	c1 = symbol_name(a)[1];
	if ('\'' == c0) return ctx;
	if (':'  == c0 && '\\' == c1) return 2;
	if (':'  == c0 && '\'' == c1) return 2;
	if (':'  == c0 && '/'  == c1) return 2;
	if ('/'  == c0) return 2;
	if (':'  == c0 && '~'  == c1) return 1;
	if (':'  == c0 && '*'  == c1) return 1;
	if ('\\' == c0 &&   0  == c1) return 2;
	if ('\\' == c0 && '~'  == c1) return 1;
	if ('\\' == c0 && '*'  == c1) return 1;
	error("internal: adverb_arity(): unknown adverb", a);
	return 0;
}

static cell monadsym(char *b) {
	char	*m = "expected monad, got";

	switch (b[0]) {
	case '!': return S_enum;
	case '#': return S_size;
	case '$': return S_format;
	case '%': return S_recip;
	case '&': return S_expand;
	case '*': return S_first;
	case '+': return S_transp;
	case ',': return S_list;
	case '-': return S_neg;
	case '<': return S_up;
	case '=': return S_group;
	case '>': return S_down;
	case '?': return S_range;
	case '@': return S_atom;
	case '^': return S_shape;
	case '_': return S_floor;
	case '|': return S_rev;
	case '~': return S_not;
	case ':':
		switch (b[1]) {
		case '#': return S_char;
		case '_': return S_undef;
		default:  error(m, make_symbol(b, 2));
		}
		break;
	}
	error("monad expected", make_symbol(b, strlen(b)));
	return UNDEFINED;
}

static cell dyadsym(char *b) {
	switch (b[0]) {
	case '!': return S_rem;
	case '#': return S_take;
	case '$': return S_format2;
	case '%': return S_div;
	case '&': return S_min;
	case '*': return S_times;
	case '+': return S_plus;
	case ',': return S_join;
	case '-': return S_minus;
	case '<': return S_lt;
	case '=': return S_eq;
	case '>': return S_gt;
	case '?': return S_find;
	case '@': return S_index;
	case '^': return S_power;
	case '_': return S_drop;
	case '|': return S_max;
	case '~': return S_match;
	case ':':
		switch (b[1]) {
		case '@': return S_indexd;
		case '#': return S_split;
		case '$': return S_form;
		case '%': return S_intdiv;
		case '+': return S_rot;
		case '-': return S_amendd;
		case ':': return S_def;
		case '=': return S_amend;
		case '^': return S_reshape;
		case '_': return S_cut;
		}
	}
	error("dyad expected", make_symbol(b, strlen(b)));
	return UNDEFINED;
}

static cell opsym(cell op, int ctx) {
	if (1 == ctx)
		return monadsym(symbol_name(op));
	else
		return dyadsym(symbol_name(op));
}

static char *opname2(cell y) {
	static char	b[100];
	if (y == S_amend) return ":=";
	if (y == S_amendd) return ":-";
	if (y == S_atom) return "@";
	if (y == S_char) return ":#";
	if (y == S_conv) return ":~";
	if (y == S_cut) return ":_";
	if (y == S_def) return "::";
	if (y == S_div) return "%";
	if (y == S_down) return ">";
	if (y == S_drop) return "_";
	if (y == S_each) return "'";
	if (y == S_each2) return "'";
	if (y == S_eachl) return ":\\";
	if (y == S_eachp) return ":'";
	if (y == S_eachr) return ":/";
	if (y == S_enum) return "~";
	if (y == S_eq) return "=";
	if (y == S_expand) return "&";
	if (y == S_find) return "?";
	if (y == S_first) return "*";
	if (y == S_floor) return "_";
	if (y == S_form) return ":$";
	if (y == S_format) return "$";
	if (y == S_format2) return "$";
	if (y == S_group) return "=";
	if (y == S_gt) return ">";
	if (y == S_if) return ":[";
	if (y == S_index) return "@";
	if (y == S_indexd) return ":@";
	if (y == S_intdiv) return ":%";
	if (y == S_iter) return ":*";
	if (y == S_join) return ",";
	if (y == S_list) return ",";
	if (y == S_lt) return "<";
	if (y == S_match) return "~";
	if (y == S_max) return "|";
	if (y == S_min) return "&";
	if (y == S_minus) return "-";
	if (y == S_neg) return "-";
	if (y == S_not) return "~";
	if (y == S_over) return "/";
	if (y == S_over2) return "/";
	if (y == S_plus) return "+";
	if (y == S_power) return "^";
	if (y == S_range) return "?";
	if (y == S_recip) return "";
	if (y == S_rem) return "!";
	if (y == S_reshape) return ":^";
	if (y == S_rev) return "|";
	if (y == S_rot) return ":+";
	if (y == S_sconv) return "\\~";
	if (y == S_siter) return "\\*";
	if (y == S_sover) return "\\";
	if (y == S_sover2) return "\\";
	if (y == S_swhile) return "\\~";
	if (y == S_shape) return "^";
	if (y == S_size) return "#";
	if (y == S_split) return ":#";
	if (y == S_take) return "#";
	if (y == S_times) return "*";
	if (y == S_transp) return "+";
	if (y == S_up) return "<";
	if (y == S_undef) return ":_";
	if (y == S_while) return ":~";
	if (y == S_x) return "x";
	if (y == S_y) return "y";
	if (y == S_z) return "z";
	strcpy(b, string(var_symbol(y)));
	return b;
}

static cell opname(cell y) {
	char	*s;

	s = opname2(y);
	return make_symbol(s, strlen(s));
}

static cell adverbsym(cell adv, int ctx) {
	int	c0, c1;

	c0 = symbol_name(adv)[0];
	c1 = symbol_name(adv)[1];
	switch (c0) {
	case '\'': return 2 == ctx? S_each2: S_each;
	case '/':  return 2 == ctx? S_over2: S_over;
	case '\\': if ('~' == c1)
			return 2 == ctx? S_swhile: S_sconv;
		   else if ('*' == c1)
			return S_siter;
		   else
			return 2 == ctx? S_sover2: S_sover;
	case ':':  switch (c1) {
			case '\\': return S_eachl;
			case '\'': return S_eachp;
			case '/':  return S_eachr;
			case '*':  return S_iter;
			case '~':  return 2 == ctx? S_while: S_conv;
		   }
		   break;
	}
	return UNDEFINED;
}

static int adverb_p(cell x) {
	int	c0, c1;

	if (!symbol_p(x))
		return 0;
	c0 = symbol_name(x)[0];
	c1 = symbol_name(x)[1];
	return	('\'' == c0 &&    0 == c1) ||
		( ':' == c0 && '\\' == c1) ||
		( ':' == c0 && '\'' == c1) ||
		( ':' == c0 &&  '/' == c1) ||
		( '/' == c0 &&    0 == c1) ||
		( ':' == c0 &&  '~' == c1) ||
		( ':' == c0 &&  '*' == c1) ||
		('\\' == c0 &&    0 == c1) ||
		('\\' == c0 &&  '~' == c1) ||
		('\\' == c0 &&  '*' == c1);
}

static int fundef_arity(cell x) {
	int	n, m;

	if (pair_p(x)) {
		if (car(x) == S_fun0 ||
		    car(x) == S_fun1 ||
		    car(x) == S_fun2 ||
		    car(x) == S_fun3
		)
			return 0;
		n = 0;
		while (x != NIL) {
			m = fundef_arity(car(x));
			if (m > n) n = m;
			x = cdr(x);
		}
		return n;
	}
	if (S_x == x) return 1;
	if (S_y == x) return 2;
	if (S_z == x) return 3;
	return 0;
}

static cell funtype(cell f) {
	int	k;

	k = fundef_arity(f);
	return 	0 == k? S_fun0:
		1 == k? S_fun1:
		2 == k? S_fun2:
		S_fun3;
}

#define syntax_error() \
	error("syntax error", Tok)

#define token() \
	kg_read()

static int cmatch(int c) {
	return symbol_p(Tok) && symbol_name(Tok)[0] == c;
}

static int cmatch2(char *c2) {
	return	symbol_p(Tok) &&
		symbol_name(Tok)[0] == c2[0] &&
		symbol_name(Tok)[1] == c2[1];
}

static void expect(int c) {
	char	b[100];

	if (cmatch(c)) {
		Tok = token();
	}
	else {
		sprintf(b, "expected :%c, got", c);
		error(b, Tok);
	}
}

static cell expr(void);
static cell prog(int fun);

static cell function(int ctx, cell *t) {
	if (cmatch('{')) {
		Tok = token();
		Infun++;
		T = prog(1);
		car(T) = funtype(T);
		Infun--;
		expect('}');
		return T;
	}
	else if (variable_p(Tok)) {
		T = Tok;
		Tok = token();
		return T;
	}
	else {
		if (t != NULL) *t = Tok;
		T = opsym(Tok, ctx);
		Tok = token();
		return T;
	}
}

static cell conditional(void) {
	cell	n;

	Incond++;
	Tok = token();
	n = cons(expr(), NIL);
	save(n);
	expect(';');
	n = cons(expr(), n);
	car(Stack) = n;
	if (cmatch2(":|")) {
		Incond--;
		n = cons(conditional(), n);
		unsave(1);
		return cons(S_if, revb(n));
	}
	else {
		expect(';');
		n = cons(expr(), n);
		car(Stack) = n;
		Incond--;
		expect(']');
		n = unsave(1);
		return cons(S_if, revb(n));
	}
}

static cell funapp_or_proj(cell v, int proj, int *fn) {
	int	n, pa;
	cell	a;

	save(v);
	pa = 0;
	Tok = token();
	a = NIL;
	save(a);
	if (cmatch(')')) {
		Tok = token();
		n = 0;
	}
	else {
		if (proj && cmatch(';')) {
			a = cons(S_x, a);
			car(Stack) = a;
			pa++;
		}
		else {
			a = cons(expr(), a);
			car(Stack) = a;
		}
		n = 1;
		if (cmatch(';')) {
			Tok = token();
			if (proj && (cmatch(';') || cmatch(')'))) {
				a = cons(pa? S_y: S_x, a);
				car(Stack) = a;
				pa++;
			}
			else {
				a = cons(expr(), a);
				car(Stack) = a;
			}
			n = 2;
			if (cmatch(';')) {
				Tok = token();
				if ( proj && cmatch(')')) {
					a = cons(pa? S_y: S_x, a);
					car(Stack) = a;
					pa++;
				}
				else {
					a = cons(expr(), a);
					car(Stack) = a;
				}
				n = 3;
			}
		}
		expect(')');
		if (pa >= n)
			error("too few arguments in projection", VOID);
	}
	v = cons(v, revb(a));
	if (pa) {
		v = cons(n>2? S_call3: n>1? S_call2: S_call1, v);
		v = cons(v, NIL);
		v = cons(pa>1? S_fun2: S_fun1, v);
	}
	else {
		v = cons(n>2? S_call3:
			 n>1? S_call2:
			 n>0? S_call1:
			 S_call0,
			 v);
	}
	if (proj && pa > 0 && (cmatch('(') || cmatch2(":("))) {
		v = funapp_or_proj(v, 0, NULL);
		pa = 0;
	}
	if (fn != NULL) *fn = pa != 0;
	unsave(2);
	return v;
}

static cell apply_adverbs(cell f, cell a1, int ctx) {
	cell	adv, n, ex;

	Tmp = f;
	save(a1);
	save(f);
	Tmp = NIL;
	adv = ex = cons(NIL, NIL);
	if (a1 != VOID)
		adv = cons(a1, adv);
	adv = cons(f, adv);
	save(adv);
	while (adverb_p(Tok)) {
		adv = cons(adverbsym(Tok, ctx), adv);
		car(Stack) = adv;
		Tok = token();
		ctx = 1;
	}
	n = expr();
	car(ex) = n;
	f = unsave(1);
	unsave(2);
	return f;
}

#define is_var(x) \
	(x == var_name(Tok)[0] && 0 == var_name(Tok)[1])

#define operator_p(x) \
	(symbol_p(x) && is_special(symbol_name(x)[0]))

static cell factor(void) {
	cell	f, v, op;
	int	fn;

	if (	number_p(Tok) ||
		char_p(Tok) ||
		string_p(Tok)
	) {
		T = Tok;
		Tok = token();
		return T;
	}
	else if (dictionary_p(Tok)) {
		T = cons(Tok, NIL);
		T = cons(S_newdict, T);
		Tok = token();
		return T;
	}
	else if (list_p(Tok)) {
		T = cons(S_lslit, Tok);
		Tok = token();
		return T;
	}
	else if (variable_p(Tok)) {
		v = Tok;
		save(v);
		if (Infun) {
			     if (is_var('x')) v = S_x;
			else if (is_var('y')) v = S_y;
			else if (is_var('z')) v = S_z;
		}
		Tok = token();
		fn = 1;
		if (cmatch('(') || cmatch2(":("))
			v = funapp_or_proj(v, 1, &fn);
		if (adverb_p(Tok)) {
			if (!fn) error("missing verb", Tok);
			v = apply_adverbs(v, VOID, 1);
		}
		unsave(1);
		return v;
	}
	else if (cmatch('(')) {
		Tok = token();
		T = expr();
		expect(')');
		return T;
	}
	else if (cmatch('{')) {
		f = function(1, &op);
		save(f);
		fn = 1;
		if (cmatch('(') || cmatch2(":("))
			f = funapp_or_proj(f, 1, &fn);
		if (adverb_p(Tok)) {
			if (!fn) error("missing verb", Tok);
			f = apply_adverbs(f, VOID, 1);
		}
		unsave(1);
		return f;
	}
	else if (cmatch2(":[")) {
		return conditional();
	}
	else if (operator_p(Tok)) {
		f = function(1, &op);
		if (adverb_p(Tok)) {
			f = opsym(op, adverb_arity(Tok, 1));
			f = apply_adverbs(f, VOID, 1);
		}
		else {
			save(f);
			f = cons(f, cons(NIL, NIL));
			car(Stack) = f;
			v = expr();
			cadr(f) = v;
			unsave(1);
		}
		return f;
	}
	else if (symbol_p(Tok)) {
		T = Tok;
		Tok = token();
		return T;
	}
	else {
		syntax_error();
		return UNDEFINED;
	}
}

#define is_delimiter(s) \
	(')' == s[0] || \
	 '}' == s[0] || \
	 ']' == s[0] || \
	 ';' == s[0] || \
	 (':' == s[0] && '|' == s[1]))

static cell expr(void) {
	cell	f, a, y, n, dy;
	cell	op;
	char	*s;
	char	name[TOKEN_LENGTH+1];
	int	fn;

	a = factor();
	while (operator_p(Tok) || variable_p(Tok) || cmatch('{')) {
		save(a);
		if (variable_p(Tok))
			s = var_name(Tok);
		else
			s = symbol_name(Tok);
		if (operator_p(Tok) && is_delimiter(s)) {
			unsave(1);
			break;
		}
		if (':' == s[0] && ':' == s[1]) {
			if (S_x == a) error("'x' is read-only", VOID);
			if (S_y == a) error("'y' is read-only", VOID);
			if (S_z == a) error("'z' is read-only", VOID);
			if (variable_p(a)) {
				if (!Infun &&
				    Module != UNDEFINED &&
				    !is_local(var_name(a)) &&
				    !is_funvar(var_name(a))
				) {
					strcpy(name, var_name(a));
					mkmodlocal(name);
					y = make_variable(name, NIL);
					a = var_symbol(y);
				}
				else {
					a = var_symbol(a);
				}
				car(Stack) = a;
			}
		}
		dy = Tok;
		save(dy);
		op = 0;
		f = function(2, &op);
		save(f);
		if ((!operator_p(dy) || '{' == symbol_name(dy)[0]) &&
		    (cmatch('(') || cmatch2(":(")))
		{
			f = funapp_or_proj(f, 1, &fn);
			car(Stack) = f;
			if (!fn) error("dyad expected", dy);
		}
		if (adverb_p(Tok)) {
			if (op) f = opsym(op, adverb_arity(Tok, 2));
			a = apply_adverbs(f, a, 2);
		}
		else {
			n = cons(expr(), NIL);
			n = cons(a, n);
			a = cons(f, n);
			if (!operator_p(dy) || '{' == symbol_name(dy)[0])
				a = cons(S_call2, a);
		}
		unsave(3);
	}
	return a;
}

static cell rename_locals(cell loc, int id) {
	cell	n, a, nn;
	char	b1[TOKEN_LENGTH], b2[TOKEN_LENGTH+1];

	if (NIL == loc)
		return NIL;
	n = cons(NIL, NIL);
	save(n);
	while (loc != NIL) {
		strcpy(b1, symbol_name(car(loc)));
		mkglobal(b1);
		sprintf(b2, "%s`%d", b1, id);
		nn = symbol_ref(b2);
		car(n) = nn;
		loc = cdr(loc);
		if (loc != NIL) {
			a = cons(NIL, NIL);
			cdr(n) = a;
			n = cdr(n);
		}
	}
	n = unsave(1);
	return n;
}

static cell prog(int fun) {
	cell	p, ps, n, mfvs, locns;
	char	*s;
	int	first = 1;

	mfvs = Mod_funvars;
	locns = Locnames;
	save(ps = NIL);
	for (;;) {
		p = expr();
		ps = cons(p, ps);
		car(Stack) = ps;
		if (!cmatch(';'))
			break;
		if (fun && first && pair_p(p) && car(p) == S_lslit) {
			if (Module != UNDEFINED)
				Mod_funvars = cons(cdr(p), Mod_funvars);
			Locnames = cons(new_atom(Local_id, cdr(p)),
					Locnames);
			cdr(p) = rename_locals(cdr(p), Local_id++);
			car(ps) = p;
		}
		first = 0;
		Tok = token();
	}
	Mod_funvars = mfvs;
	Locnames = locns;
	car(Stack) = ps = revb(ps);
	if (	fun &&
		ps != NIL &&
		pair_p(car(ps)) &&
		S_lslit == caar(ps) &&
		cdr(ps) != NIL)
	{
		for (p = car(ps); p != NIL; p = cdr(p)) {
			if (symbol_p(car(p))) {
				s = symbol_name(car(p));
				n = make_variable(s, NO_VALUE);
				car(p) = n;
			}
		}
	}
	return cons(S_prog, unsave(1));
}

static cell parse(char *p) {
	cell	x;

	open_input_string(p);
	Tok = token();
	if (END_OF_FILE == Tok) {
		close_input_string();
		return END_OF_FILE;
	}
	x = prog(0);
	if (Tok != END_OF_FILE)
		syntax_error();
	close_input_string();
	return x;
}

/*
 * Bytecode Compiler, Code generator
 */

static void emit(cell p) {
	save(p);
	if (NIL == P) {
		P = Prog = cons(p, NIL);
	}
	else {
		cdr(P) = cons(p, NIL);
		P = cdr(P);
	}
	unsave(1);
}

static void	comp(cell p);

static void comp_funcall(cell p, int k) {
	cell	q;
	int	i;

	if (length(cddr(p)) != k)
		error("wrong argument count", cdr(p));
	for (i=0, q = cddr(p); i<k; q = cdr(q), i++)
		comp(car(q));
	comp(cadr(p));
	emit(3==k? S_call3: 2==k? S_call2: 1==k? S_call1: S_call0);
	emit(3==k? S_pop3: 2==k? S_pop2: 1==k? S_pop1: S_pop0);
}

static void comp_fundef(cell p, int k) {
	cell	q, f;

	f = P;
	for (q = cdr(p); q != NIL; q = cdr(q)) {
		comp(car(q));
		if (cdr(q) != NIL)
			emit(S_clear);
	}
	P = f; emit(NIL == f? Prog: cdr(f));
	emit(3==k? S_fun3: 2==k? S_fun2: 1==k? S_fun1: S_fun0);
}

#define adverb_op_p(op) \
	(op) == S_conv || (op) == S_each || (op) == S_sconv || \
	(op) == S_iter || (op) == S_siter || (op) == S_while || \
	(op) == S_swhile || (op) == S_eachp || (op) == S_over || \
	(op) == S_sover || (op) == S_each2 || (op) == S_eachl || \
	(op) == S_eachr || (op) == S_over2 || (op) == S_sover2

static int adverb_op_arity(cell op) {
	if (op == S_conv) return 1;
	if (op == S_each) return 1;
	if (op == S_sconv) return 1;
	if (op == S_iter) return 1;
	if (op == S_siter) return 1;
	if (op == S_while) return 1;
	if (op == S_swhile) return 1;
	if (op == S_eachp) return 2;
	if (op == S_over) return 2;
	if (op == S_sover) return 2;
	if (op == S_each2) return 2;
	if (op == S_eachl) return 2;
	if (op == S_eachr) return 2;
	if (op == S_over2) return 2;
	if (op == S_sover2) return 2;
	error("internal: adverb_op_arity():  bad adverb op", op);
	return 0;
}

static int adverb_op_ctx(cell op) {
	if (op == S_conv) return 1;
	if (op == S_each) return 1;
	if (op == S_sconv) return 1;
	if (op == S_iter) return 2;
	if (op == S_siter) return 2;
	if (op == S_while) return 2;
	if (op == S_swhile) return 2;
	if (op == S_eachp) return 1;
	if (op == S_over) return 1;
	if (op == S_sover) return 1;
	if (op == S_each2) return 2;
	if (op == S_eachl) return 2;
	if (op == S_eachr) return 2;
	if (op == S_over2) return 2;
	if (op == S_sover2) return 2;
	error("internal: adverb_op_ctx():  bad adverb op", op);
	return 0;
}

static void comp_adverb(cell p, int args) {
	cell	f, q;
	int	ctx, aa, nest;

	aa = adverb_op_arity(car(p));
	if (args) {
		f = p;
		for (q = p; adverb_op_p(car(q)); q = cdr(q))
			f = q;
		ctx = adverb_op_ctx(car(f));
		if (2 == ctx) {
			if (NIL == cdr(q) || NIL == cddr(q) || cdddr(q) != NIL)
				error("wrong adverb context", opname(car(f)));
			comp(caddr(q));
			comp(cadr(q));
			emit(S_swap);
		}
		else {
			if (NIL == cdr(q) || cddr(q) != NIL)
				error("wrong adverb context", car(f));
			comp(cadr(q));
		}
	}
	nest = 0;
	if (adverb_op_p(cadr(p))) {
		f = P;
		comp_adverb(cdr(p), 0);
		P = f; emit(cdr(f));
		nest = 1;
	}
	if (nest && aa > 1)
		error("monad expected in chained adverb", opname(car(p)));
	if (	variable_p(cadr(p)) &&
		cadr(p) != S_x &&
		cadr(p) != S_y &&
		cadr(p) != S_z &&
		'%' == var_name(cadr(p))[0])
	{
		if (!nest) emit(cons(cadr(p), NIL));
		emit(2==aa? S_imm2: S_imm1);
	}
	else {
		f = P;
		comp(cadr(p));
		emit(2==aa? S_call2: S_call1);
		emit(2==aa? S_pop2: S_pop1);
		P = f; emit(cdr(f));
		emit(2==aa? S_imm2: S_imm1);
	}
	emit(car(p));
}

static void comp(cell p) {
	cell	op, q;

	if (!atom_p(p))
		op = car(p);
	else
		op = UNDEFINED;
	if (atom_p(p)) {
		emit(p);
	}
	else if (op == S_lslit) {
		emit(cdr(p));
	}
	else if (op == S_newdict) {
		emit(cadr(p));
		emit(S_newdict);
	}
	else if (op == S_atom || op == S_char || op == S_down ||
		 op == S_enum || op == S_expand || op == S_first ||
		 op == S_floor || op == S_format || op == S_group ||
		 op == S_list || op == S_neg || op == S_not ||
		 op == S_range || op == S_recip || op == S_rev ||
		 op == S_shape || op == S_size || op == S_transp ||
		 op == S_up || op == S_undef)
	{
		if (NIL == cdr(p) || cddr(p) != NIL)
			error("wrong argument count", op);
		comp(cadr(p));
		emit(car(p));
	}
	else if (op == S_amend || op == S_amendd || op == S_cut ||
		 op == S_def || op == S_div || op == S_drop ||
		 op == S_eq || op == S_find || op == S_form ||
		 op == S_format2 || op == S_gt || op == S_index ||
		 op == S_indexd || op == S_intdiv || op == S_join ||
		 op == S_lt || op == S_match || op == S_max ||
		 op == S_min || op == S_minus || op == S_plus ||
		 op == S_power || op == S_rem || op == S_reshape ||
		 op == S_rot || op == S_split || op == S_take ||
		 op == S_times)
	{
		if (NIL == cdr(p) || NIL == cddr(p) || cdddr(p) != NIL)
			error("wrong argument count", op);
		comp(caddr(p));
		comp(cadr(p));
		emit(S_swap);
		emit(car(p));
	}
	else if (op == S_conv || op == S_each || op == S_sconv ||
		 op == S_iter || op == S_siter || op == S_while ||
		 op == S_swhile || op == S_eachp || op == S_over ||
		 op == S_sover || op == S_each2 || op == S_eachl ||
		 op == S_eachr || op == S_over2 || op == S_sover2)
	{
		comp_adverb(p, 1);
	}
	else if (op == S_prog) {
		for (p = cdr(p); p != NIL; p = cdr(p)) {
			comp(car(p));
			if (cdr(p) != NIL)
				emit(S_clear);
		}
	}
	else if (op == S_if) {
		comp(cadr(p));
		q = P; comp(caddr(p));
		P = q; emit(cdr(q));
		q = P; comp(cadddr(p));
		P = q; emit(cdr(q));
		emit(S_if);
	}
	else if (op == S_call0)  { comp_funcall(p, 0); }
	else if (op == S_call1)  { comp_funcall(p, 1); }
	else if (op == S_call2)  { comp_funcall(p, 2); }
	else if (op == S_call3)  { comp_funcall(p, 3); }
	else if (op == S_fun0)   { comp_fundef(p, 0); }
	else if (op == S_fun1)   { comp_fundef(p, 1); }
	else if (op == S_fun2)   { comp_fundef(p, 2); }
	else if (op == S_fun3)   { comp_fundef(p, 3); }
	else {
		error("internal: unknown operator in AST", p);
	}
}

static cell compile(char *s) {
	cell	p;

	p = parse(s);
	if (END_OF_FILE == p)
		return END_OF_FILE;
	save(p);
	P = Prog = NIL;
	comp(p);
	if (Debug) {
		kg_write(Prog);
		nl();
	}
	unsave(1);
	return Prog;
}

/*
 * Interpreters
 */

static cell pjoin(cell a, cell b) {
	Tmp = b;
	save(a);
	save(b);
	Tmp = NIL;
	a = join(a, b);
	unsave(2);
	return a;
}

static cell load(cell x, int v, int scr) {
	int	p, oldp, oline, oldprog;
	char	*s, *kp, kpbuf[TOKEN_LENGTH+1];
	cell	n = NIL; /*LINT*/
	cell	r = NIL;

	save(x);
	if ('.' == string(x)[0] || '/' == string(x)[0]) {
		save(NIL);
		n = x;
		p = open_input_port(string(x));
		if (p < 0) {
			n = pjoin(x, make_string(".kg", 3));
			car(Stack) = n;
			p = open_input_port(string(n));
		}
	}
	else {
		s = getenv("KLONGPATH");
		if (NULL == s) {
			strcpy(kpbuf, DFLPATH);
		}
		else {
			strncpy(kpbuf, s, TOKEN_LENGTH);
			kpbuf[TOKEN_LENGTH] = 0;
			if (strlen(kpbuf) >= TOKEN_LENGTH) {
				error("KLONGPATH too long!", VOID);
				return UNDEFINED;
		}
		}
		kp = strtok(kpbuf, ":");
		p = -1;
		save(NIL);
		while (kp != NULL) {
			n = pjoin(make_string("/", 1), x);
			car(Stack) = n;
			n = pjoin(make_string(kp, strlen(kp)), n);
			if ((p = open_input_port(string(n))) >= 0)
				break;
			n = pjoin(n, make_string(".kg", 3));
			if ((p = open_input_port(string(n))) >= 0)
				break;
			kp = strtok(NULL, ":");
		}
	}
	if (p < 0) {
		error(".l: cannot open file", x);
		unsave(2);
		return UNDEFINED;
	}
	if (1 == v && 0 == Quiet && NIL == Loading) {
		prints("loading ");
		kg_write(n);
		nl();
	}
	close_input_string();
	lock_port(p);
	oldp = input_port();
	set_input_port(p);
	oline = Line;
	Line = 1;
	Loading = cons(x, Loading);
	oldprog = Prog_chan;
	Prog_chan = p;
	save(r);
	if (scr) kg_getline(kpbuf, TOKEN_LENGTH);
	for (;;) {
		Tok = token();
		if (port_eof(p))
			break;
		if (END_OF_FILE == Tok)
			continue;
		x = prog(0);
		if (Tok != END_OF_FILE)
			syntax_error();
		save(x);
		P = Prog = 0;
		comp(x);
		x = Prog;
		unsave(1);
		if (s9_aborted())
			break;
		set_input_port(oldp);
		eval(x);
		r = car(Stack) = car(Dstack);
		set_input_port(p);
		op_clear();
	}
	Prog_chan = oldprog;
	Loading = cdr(Loading);
	Line = oline;
	set_input_port(oldp);
	unlock_port(p);
	close_port(p);
	unsave(3);
	return r;
}

static cell evalstr(cell x) {
	char	buf[TOKEN_LENGTH+1];

	if (string_len(x) >= TOKEN_LENGTH)
		return error("evalstr: expression too long", VOID);
	strcpy(buf, string(x));
	x = compile(buf);
	if (END_OF_FILE == x)
		return END_OF_FILE;
	if (s9_aborted())
		return UNDEFINED;
	eval(x);
	if (s9_aborted())
		return UNDEFINED;
	return pop();
}

static void eval_arg(char *s, int echo) {
	cell	x;

	x = make_string(s, strlen(s));
	save(x);
	x = evalstr(x);
	unsave(1);
	if (s9_aborted())
		bye(1);
	if (0 == echo)
		return;
	kg_write(x);
	nl();
}

static void interpret(void) {
	cell	x;

	for (;;) {
		Prog_chan = 0;
		reset_std_ports();
		Dstack = NIL;
		s9_reset();
		Intr = 0;
		if (!Quiet) {
			prints("        ");
			flush();
		}
		if (kg_getline(Inbuf, TOKEN_LENGTH) == NULL && Intr == 0)
			break;
		if (Intr)
			continue;
		x = compile(Inbuf);
		transcribe(make_string(Inbuf, strlen(Inbuf)), 1);
		if (s9_aborted() || atom_p(x))
			continue;
		Safe_dict = Sys_dict;
		eval(x);
		if (s9_aborted()) {
			Sys_dict = Safe_dict;
			continue;
		}
		set_output_port(1);
		kg_write(car(Dstack));
		nl();
		transcribe(car(Dstack), 0);
		var_value(S_it) = car(Dstack);
	}
}

static void make_image_file(char *s) {
	char	magic[16];
	char	errbuf[128];
	char	*r;

	memcpy(magic, "KLONGYYYYMMDD___", 16);
	memcpy(&magic[5], VERSION, 8);
	image_vars(Image_vars);
	r = dump_image(s, magic);
	if (NULL == r)
		return;
	sprintf(errbuf, "kg: dump_image(): %s", r);
	fatal(errbuf);
}

static void load_image_file(void) {
	char	magic[16];
	char	kpbuf[TOKEN_LENGTH+1];
	char	errbuf[128];
	char	*r, *kp;
	FILE	*f;

	memcpy(magic, "KLONGYYYYMMDD___", 16);
	memcpy(&magic[5], VERSION, 8);
	if (NULL == (kp = getenv("KLONGPATH")))
		kp = DFLPATH;
	if (strlen(kp) >= TOKEN_LENGTH)
		fatal("KLONGPATH too long");
	strcpy(kpbuf, kp);
	kp = strtok(kpbuf, ":");
	while (kp != NULL) {
		sprintf(Image_path, "%s/klong.image", kp);
		f = fopen(Image_path, "r");
		if (f != NULL) {
			r = load_image(Image_path, magic);
			if (r != NULL) {
				fprintf(stderr, "kg: bad image file: %s\n",
					Image_path);
				sprintf(errbuf, "load_image(): %s", r);
				fatal(errbuf);
			}
		}
		kp = strtok(NULL, ":");
	}
}

/*
 * Initialization and Startup
 */

static void init(void) {
	int	i;
	cell	n, op;

	cleartrace();
	s9_init(GC_root, NULL, NULL);
	image_vars(Image_vars);
	Dstack = NIL;
	Frame = NIL;
	Locals = NIL;
	Quiet = 0;
	Script = 0;
	Sys_dict = NIL;
	From_chan = 0;
	To_chan = 1;
	Prog_chan = 0;
	Line = 1;
	Listlev = 0;
	Incond = 0;
	Report = 1;
	Transcript = -1;
	Infun = 0;
	Loading = NIL;
	Module = UNDEFINED;
	Locnames = NIL;
	Local_id = 0;
	Display = 0;
	Intr = 0;
	Image_path[0] = 0;
	Barrier = new_atom(T_BARRIER, 0);
	srand(time(NULL)*123);
	for (i = 0; Ops[i].name != NULL; i++) {
		op = make_primop(i, Ops[i].syntax);
		make_variable(Ops[i].name, op);
	}
	S_amend = var_ref("%amend");
	S_amendd = var_ref("%amendd");
	S_atom = var_ref("%atom");
	S_call0 = var_ref("%call0");
	S_call1 = var_ref("%call1");
	S_call2 = var_ref("%call2");
	S_call3 = var_ref("%call3");
	S_char = var_ref("%char");
	S_clear = var_ref("%clear");
	S_conv = var_ref("%conv");
	S_cut = var_ref("%cut");
	S_def = var_ref("%def");
	S_div = var_ref("%div");
	S_down = var_ref("%down");
	S_drop = var_ref("%drop");
	S_each = var_ref("%each");
	S_each2 = var_ref("%each2");
	S_eachl = var_ref("%eachl");
	S_eachp = var_ref("%eachp");
	S_eachr = var_ref("%eachr");
	S_enum = var_ref("%enum");
	S_eq = var_ref("%eq");
	S_expand = var_ref("%expand");
	S_find = var_ref("%find");
	S_first = var_ref("%first");
	S_floor = var_ref("%floor");
	S_form = var_ref("%form");
	S_format = var_ref("%format");
	S_format2 = var_ref("%format2");
	S_fun0 = var_ref("%fun0");
	S_fun1 = var_ref("%fun1");
	S_fun2 = var_ref("%fun2");
	S_fun3 = var_ref("%fun3");
	S_group = var_ref("%group");
	S_gt = var_ref("%gt");
	S_if = var_ref("%if");
	S_imm1 = var_ref("%imm1");
	S_imm2 = var_ref("%imm2");
	S_index = var_ref("%index");
	S_indexd = var_ref("%indexd");
	S_intdiv = var_ref("%intdiv");
	S_it = var_ref("it");
	S_iter = var_ref("%iter");
	S_join = var_ref("%join");
	S_list = var_ref("%list");
	S_lslit = var_ref("%lslit");
	S_lt = var_ref("%lt");
	S_match = var_ref("%match");
	S_max = var_ref("%max");
	S_min = var_ref("%min");
	S_minus = var_ref("%minus");
	S_neg = var_ref("%neg");
	S_newdict = var_ref("%newdict");
	S_not = var_ref("%not");
	S_over = var_ref("%over");
	S_over2 = var_ref("%over2");
	S_plus = var_ref("%plus");
	S_pop0 = var_ref("%pop0");
	S_pop1 = var_ref("%pop1");
	S_pop2 = var_ref("%pop2");
	S_pop3 = var_ref("%pop3");
	S_power = var_ref("%power");
	S_prog = var_ref("%prog");
	S_range = var_ref("%range");
	S_recip = var_ref("%recip");
	S_rem = var_ref("%rem");
	S_reshape = var_ref("%reshape");
	S_rev = var_ref("%rev");
	S_rot = var_ref("%rot");
	S_sconv = var_ref("%sconv");
	S_siter = var_ref("%siter");
	S_sover = var_ref("%sover");
	S_sover2 = var_ref("%sover2");
	S_swhile = var_ref("%swhile");
	S_shape = var_ref("%shape");
	S_size = var_ref("%size");
	S_split = var_ref("%split");
	S_swap = var_ref("%swap");
	S_syscall = var_ref("%syscall");
	S_take = var_ref("%take");
	S_times = var_ref("%times");
	S_transp = var_ref("%transp");
	S_up = var_ref("%up");
	S_undef = var_ref("%undef");
	S_while = var_ref("%while");
	S_x = var_ref("%x");
	S_y = var_ref("%y");
	S_z = var_ref("%z");
	S_argv = var_ref(".a");
	n = var_ref(".cin");
	var_value(n) = make_port(0, T_INPUT_PORT);
	n = var_ref(".cout");
	var_value(n) = make_port(1, T_OUTPUT_PORT);
	n = var_ref(".cerr");
	var_value(n) = make_port(2, T_OUTPUT_PORT);
	Epsilon_var = make_variable(".e", Epsilon);
	S_thisfn = var_ref(".f");
	S_host = var_ref(".h");
#ifdef plan9
	n = symbol_ref("plan9");
#else
	n = symbol_ref("unix");
#endif
	var_value(S_host) = n;
	n = var_ref(".helpdb");
	var_value(n) = NIL;
	for (i = 0; Sysfns[i].name != NULL; i++) {
		n = cons(S_syscall, NIL);
		save(n);
		n = make_integer(i);
		n = cons(n, unsave(1));
		op = make_function(n, 0, Sysfns[i].arity);
		make_variable(Sysfns[i].name, op);
	}
}

#ifdef plan9
 void keyboard_interrupt(void *dummy, char *note) {
	USED(dummy);
	if (strstr(note, "interrupt") == NULL)
		noted(NDFLT);
	reset_std_ports();
	error("interrupted", VOID);
	Intr = 1;
	noted(NCONT);
 }
#else
 void keyboard_interrupt(int sig) {
	reset_std_ports();
	error("interrupted", VOID);
	Intr = 1;
	handle_sigint();
 }

 void keyboard_quit(int sig) {
	fatal("quit signal received, exiting");
 }
#endif

static void usage(int x) {
	prints("Usage: kg [-dhnqsuv?] [-e expr] [-l file] [-o file]");
	nl();
	prints("          [-r expr] [-t file] [file [args]] [-- args]");
	nl();
	if (x) {
		bye(1);
	}
}

static void longusage(void) {
	char	*s;

	nl();
	prints("Klong ");
	prints(VERSION);
	prints(" by Nils M Holm, in the public domain");
	nl();
	nl();
	usage(0);
	nl();
	prints("-d         debug (print bytecode and call traces)"); nl();
	prints("-e expr    evaluate expression, no interactive mode"); nl();
	prints("-l file    load program from file"); nl();
	prints("-n         clean start: don't parse KLONGOPTS, don't"); nl();
	prints("           load any image (must be first option!)"); nl();
	prints("-o file    output image to file, then exit"); nl();
	prints("-q         quiet (no banner, no prompt, exit on errors)");
	nl();
	prints("-r expr    run expr (like -e, but don't print result)"); nl();
	prints("-s         skip first line (#!) in script files"); nl();
	prints("-t file    send transcript to file"); nl();
	prints("-u         allocate unlimited memory (use with care!)"); nl();
	prints("file args  run program with arguments, then exit"); nl();
	prints("-- args    bind remaining arguments to .a"); nl();
	nl();
	if ((s = getenv("KLONGPATH")) != NULL) {
		prints("KLONGPATH = ");
		prints(s);
		nl();
	}
	if ((s = getenv("KLONGOPTS")) != NULL) {
		prints("KLONGOPTS = ");
		prints(s);
		nl();
	}
	if (Image_path[0] != 0) {
		prints("Imagefile = ");
		prints(Image_path);
		nl();
	}
	nl();
	bye(0);
}

static cell	New;

#define setargv(a) \
	do { \
		New = argv_to_list(&argv[a]); \
		var_value(S_argv) = New; \
	} while (0)

static int readopts(int argc, char **argv, int *p_loop, int *p_endargs) {
	int	i, j, echo, loop = 1, endargs = 0;

	for (i=1; i<argc; i++) {
		if (argv[i][0] != '-')
			break;
		for (j=1; argv[i][j]; j++) {
			switch (argv[i][j]) {
			case '?':
			case 'v':
			case 'h':	longusage();
					break;
			case 'd':	Debug = 1;
					break;
			case 'r':
			case 'e':	echo = argv[i][j] == 'e';
					if (++i >= argc)
						usage(1);
					Quiet = 1;
					setargv(i+1);
					eval_arg(argv[i], echo);
					j = strlen(argv[i])-1;
					loop = 0;
					break;
			case 'l':	if (++i >= argc)
						usage(1);
					load(make_string(argv[i],
							strlen(argv[i])),
						0, 0);
					j = strlen(argv[i])-1;
					break;
			case 'n':	fprintf(stderr, "kg: -n must be"
							" first option\n");
					bye(1);
			case 'o':	if (++i >= argc)
						usage(1);
					make_image_file(argv[i]);
					bye(0);
					break;
			case 'q':	Quiet = 1;
					break;
			case 's':	Script = 1;
					break;
			case 't':	if (++i >= argc)
						usage(1);
					transcript(argv[i]);
					j = strlen(argv[i])-1;
					break;
			case 'u':	set_node_limit(0);
					set_vector_limit(0);
					break;
			case '-':	endargs = 1;
					break;
			default:	usage(1);
					break;
			}
		}
		if (endargs) {
			i++;
			break;
		}
	}
	if (p_endargs != NULL && p_loop != NULL) {
		*p_endargs = endargs;
		*p_loop = loop;
	}
	return i;
}

static void klongopts(void) {
	#define MAX 100
	char	*a[MAX], *k, *kp;
	int	i;

	if ((k = getenv("KLONGOPTS")) == NULL)
		return;
	k = strdup(k);
	kp = strtok(k, " ");
	a[0] = "";
	i = 1;
	while (kp != NULL) {
		if (i >= MAX-1)
			fatal("too many KLONGOPTS arguments");
		a[i++] = kp;
		kp = strtok(NULL, " ");
	}
	a[i] = NULL;
	readopts(i, a, NULL, NULL);
	free(k);
}

int main(int argc, char **argv) {
	int	i, loop, endargs;

	init();
	if (!(argc > 1 && '-' == argv[1][0] && 'n' == argv[1][1])) {
		klongopts();
		load_image_file();
		i = readopts(argc, argv, &loop, &endargs);
	}
	else {
		i = readopts(argc-1, argv+1, &loop, &endargs) + 1;
	}
	if (endargs == 0 && i < argc) {
		Quiet = 1;
		setargv(i+1);
		load(make_string(argv[i], strlen(argv[i])), 0, Script);
		bye(0);
	}
	if (!Quiet) {
		handle_sigint();
		handle_sigquit();
		prints("        Klong ");
		prints(VERSION);
		nl();
	}
	if (loop) {
		setargv(i);
		interpret();
	}
	if (!Quiet) prints("bye!\n");
	bye(0);
	return 0;
}
