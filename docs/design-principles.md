# Design Principles

This page states the rules that decide what goes into the Klong language as KlongPy implements it. Read it before proposing a change to the language, and use it to judge one.

The principles are in priority order. When two of them disagree, the earlier one decides.

## 1. Arrays are the point

Klong expresses a computation over whole arrays. An addition must make that style easier to write or to read.

The language has no loop statement and does not gain one. Iteration is an adverb applied to a function. A feature that pulls programs toward handling one element at a time does not belong.

## 2. One language on every target

A language feature means the same thing in the interpreter, in compiled C and on a microcontroller.

A feature that cannot be given the same meaning everywhere is a library function of the host, and is named as one. Reading a web page, querying a database and calling Python are library functions. Function definition, assignment and name lookup are language, and they behave the same on every target.

## 3. Reuse before inventing

A new capability is built from what the language has: lists, dictionaries, functions and symbols. A new concept is added only when those cannot express it, and new syntax is the last resort.

A dictionary written from expressions, `:{;k;v}`, is still a dictionary. A function with named parameters is still a function, and adverbs, projections and IPC treat it as one.

## 4. No hidden state

A program says where its values come from. Prefer a value that can be seen and passed to a function over a mode that changes what later lines mean, and prefer an argument over a name read from whoever called.

## 5. Mistakes are loud

A program that is probably wrong says so, with the position of the problem. An error names what failed and, where one exists, the form that was meant.

A silent wrong answer is a defect in the language or the implementation. It is not the user's mistake.

## 6. Readable without knowing APL

Where a terse form and a named form say the same thing, the named form exists and is the one the documentation teaches. `area(w;h)::{w*h}` is taught before `{x*y}`.

The terse forms stay. Short functions given to adverbs are clearer in them, and every Klong program uses them.

## 7. Classic Klong still runs

An addition takes a meaning that was an error or did nothing. It does not change what a valid Klong program does.

This principle is a default and not a constraint. It gives way when it blocks one of the principles above it. A change that breaks a valid Klong program is listed in [Migrating to v2](migrating-to-v2.md) with the reason, and [Differences from Klong](differences.md) records where KlongPy and Klong part.

## Using the principles

For a proposed change, answer these questions in order. The first answer that is "no" settles it.

1. Does it keep array style the natural way to write the program?
2. Can it mean the same thing on every target? If not, it is a library function.
3. Can it be built from lists, dictionaries, functions and symbols, without new syntax?
4. Is every value it uses visible where it is used?
5. Does a mistake in using it produce an error with a position?
6. Can someone who has not used an array language read it?
7. Does every valid Klong program still do what it did? If not, is the reason one of the questions above?
