/*
 * Copyright (c) 2014, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the uvwsim library.
 * Contact: uvwsim at oerc.ox.ac.uk
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PYUVWSIM_H_
#define PYUVWSIM_H_

/* Platform recognition */
#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
#    define PYUVWSIM_OS_WIN32
#endif
#if (defined(WIN64) || defined(_WIN64) || defined(__WIN64__))
#    define PYUVWSIM_OS_WIN64
#endif
#if (defined(PYUVWSIM_OS_WIN32) || defined(PYUVWSIM_OS_WIN64))
#    define PYUVWSIM_OS_WIN
#endif

/* Declare import and export macros */
#ifndef PYUVWSIM_DECL_EXPORT
#   ifdef PYUVWSIM_OS_WIN
#       define PYUVWSIM_DECL_EXPORT __declspec(dllexport)
#   elif __GNUC__ >= 4
#       define PYUVWSIM_DECL_EXPORT __attribute__ ((visibility("default")))
#   else
#       define PYUVWSIM_DECL_EXPORT
#   endif
#endif
#ifndef PYUVWSIM_DECL_IMPORT
#   ifdef PYUVWSIM_OS_WIN
#       define PYUVWSIM_DECL_IMPORT __declspec(dllimport)
#   elif __GNUC__ >= 4
#       define PYUVWSIM_DECL_IMPORT __attribute__ ((visibility("default")))
#   else
#       define PYUVWSIM_DECL_IMPORT
#   endif
#endif

/* PYUVWSIM_API is used to identify public functions which should be exported */
#ifdef pyuvwsim_EXPORTS
#   define PYUVWSIM_API PYUVWSIM_DECL_EXPORT
#else
#   define PYUVWSIM_API PYUVWSIM_DECL_IMPORT
#endif


#endif /* PYUVWSIM_H_ */
