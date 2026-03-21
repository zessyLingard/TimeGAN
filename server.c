/*
 * CCEAP: An Implementation of an Educational Protocol for Covert Channel
 *        Analysis Purposes
 *
 * The goal of this tool is to provide a simple, accessible implementation
 * for students. The tool demonstrates several network covert channel
 * vulnerabilities in a single communication protocol.
 *
 * Copyright (C) 2016-2019 Steffen Wendzel, steffen (at) wendzel (dot) de
 *                    https://www.wendzel.de
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Cf. `LICENSE' file.
 *
 */

#include "main.h"

#ifdef __MACH__
 #include <mach/clock.h>
 #include <mach/mach.h>
#endif

void print_time_diff(FILE *output_file)
{
	long ns;
	time_t s;
	struct timespec spec_now;
	static struct timespec spec_last;
	static int first_call = 1;
#ifdef __MACH__ /* code from Stackoverflow.com (Mac OS lacks clock_gettime()) */
	#warning "Including experimental code for MacOS. CCEAP runs best on Linux!"
	clock_serv_t cclock;
	mach_timespec_t mts;

	host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
	clock_get_time(cclock, &mts);
	mach_port_deallocate(mach_task_self(), cclock);
	spec_now.tv_sec = mts.tv_sec;
	spec_now.tv_nsec = mts.tv_nsec;
#else
	clock_gettime(CLOCK_REALTIME, &spec_now);
#endif
	
	if (first_call) {
		printf("0.000000000\n");
		if (output_file) fprintf(output_file, "0.000000000\n");
		first_call = 0;
	} else {
		s  = spec_now.tv_sec - spec_last.tv_sec;
		ns = spec_now.tv_nsec - spec_last.tv_nsec;
		
		if (ns < 0) {
			ns = 1000000000 + ns; // 1 billion nanoseconds = 1 second
			s -= 1;
		}
		
		// Calculate microseconds for display
		long us = ns / 1000;  // Convert nanoseconds to microseconds
		
		// Print with standard precision for console (milliseconds)
		printf("%"PRIdMAX".%03ld\n", (intmax_t)s, us / 1000);
		
		// For the file output, format with microsecond precision (9 digits after decimal)
		if (output_file) {
			// Format with exactly 9 decimal places (0.123456789 format)
			double seconds = (double)s + (double)ns / 1000000000.0;
			fprintf(output_file, "%.9f\n", seconds);
		}
	}
	bcopy(&spec_now, &spec_last, sizeof(struct timespec));
}

void
inform_disconnected(FILE *times_file)
{
	fprintf(stderr, "The connection was closed by the foreign host.\n");
	fprintf(stderr, "Shutdown ..\n");
	
	// Close the timing file if it's open
	if (times_file) {
		fclose(times_file);
		fprintf(stderr, "Timing file closed.\n");
	}
	
	exit(OK_EXIT);
}

int
main(int argc, char *argv[])
{
	int ch;
	int sockfd;
	int connfd;
	socklen_t salen = sizeof(struct sockaddr_in);
	int yup;
	struct sockaddr_in sa;
	cceap_header_t *hdr;
	int lst_port = 0;
	options_t *options_array;
	int verbose = 0;
	int quiet = 0;
	int max = 0;
	int ret = 0;
	int x;
	fd_set rset;
	char buf[4096]; // Increased buffer size from 1024 to 4096
	FILE *times_file = NULL;
	char *times_filepath = NULL;
	
	// Variables for packet reassembly
	static char packet_buffer[8192] = {0}; // Buffer for reassembling fragmented packets
	static int packet_buffer_size = 0;     // Current size of data in reassembly buffer
	
	while ((ch = getopt(argc, argv, "vhP:qT:")) != -1) {
		switch (ch) {
		case 'v':
			verbose = 1;
			break;
		case 'P':
			/* TCP port to listen on */
			lst_port = atoi(optarg);
			break;
		case 'q':
			quiet = 1;
			break;
		case 'T':
			/* File to output timing information */
			times_filepath = optarg;
			break;
		case 'h':
		default:
			usage(USAGE_SERVER);
			/* NOTREACHED */
		}
	}

	if (!quiet)
		print_gpl();

	/* basic checks of provided parameters */
	if (lst_port >= 0xffff || lst_port < 1) {
		fprintf(stderr, "TCP listen port out of range or not specified.\n");
		exit(ERR_EXIT);
	}
	
	/* short welcome notice */
	if (quiet) {
		fprintf(stdout, "---\n");
	} else {
		fprintf(stdout, "CCEAP - Covert Channel Educational Analysis Protocol (Server)\n");
		fprintf(stdout, "   => version: " CCEAP_VER ", written by: " CCEAP_AUTHOR "\n");
	}
	
	/* summarize information if verbose mode is used */
	if (verbose) {
		printf("verbose output:\n");
		printf("* Using TCP port %d\n", lst_port);
		if (times_filepath) {
			printf("* Writing timing information to: %s\n", times_filepath);
		}
	}
	
	/* Open the timing output file if specified */
	if (times_filepath) {
		times_file = fopen(times_filepath, "w");
		if (!times_file) {
			err(ERR_EXIT, "Failed to open timing output file: %s", times_filepath);
		}
		if (verbose) {
			printf("* Timing output file opened successfully\n");
		}
	}
	
	if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
		err(ERR_EXIT, "socket()");

	if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yup, sizeof(yup)) != 0)
		err(ERR_EXIT, "setsockopt(..., SO_REUSEADDR, ...)");
	
	// Set TCP_NODELAY to disable Nagle's algorithm
	int flag = 1;
	if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) != 0)
		err(ERR_EXIT, "setsockopt(..., TCP_NODELAY, ...)");
	
	// Increase socket buffer sizes
	int rcvbuf = 262144; // 256KB
	if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) != 0)
		err(ERR_EXIT, "setsockopt(..., SO_RCVBUF, ...)");
	
	/* prepare and create the connection to the peer */
	bzero(&sa, sizeof(struct sockaddr_in));
	sa.sin_family = AF_INET;
	sa.sin_port = htons(lst_port);
	sa.sin_addr.s_addr = INADDR_ANY;
	
	if (bind(sockfd, (struct sockaddr *) &sa, sizeof(struct sockaddr_in)) < 0)
		err(ERR_EXIT, "bind()"); /* socket for client connection */
	
	if (listen(sockfd, 1) < 0)
		err(ERR_EXIT, "listen()");
	
	if ((connfd = accept(sockfd, (struct sockaddr *) &sa, &salen)) < 0) {
		err(ERR_EXIT, "accept()");
	}
	
	// Set TCP_NODELAY for the connection socket too
	if (setsockopt(connfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag)) != 0)
		err(ERR_EXIT, "setsockopt(..., TCP_NODELAY, ...)");
	
	// Increase socket buffer sizes for the connection socket too
	if (setsockopt(connfd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf)) != 0)
		err(ERR_EXIT, "setsockopt(..., SO_RCVBUF, ...)");
	
	max = connfd;
	for (;;) {
		FD_ZERO(&rset);
		FD_SET(connfd, &rset);
		
		ret = select(max + 1, &rset, NULL, NULL, NULL);
		if (ret) {
			int size_header = sizeof(cceap_header_t);
		
			bzero(buf, sizeof(buf));

			if (FD_ISSET(connfd, &rset)) {
				if ((x = recv(connfd, buf, sizeof(buf) - 1, 0)) == -1)
					err(ERR_EXIT, "recv()");
				if (x == 0)
					inform_disconnected(times_file);
				
				// Add received data to our packet buffer
				if (packet_buffer_size + x <= sizeof(packet_buffer)) {
					memcpy(packet_buffer + packet_buffer_size, buf, x);
					packet_buffer_size += x;
				} else {
					fprintf(stderr, "Packet buffer overflow, clearing buffer\n");
					packet_buffer_size = 0;
				}
				
				// Process as many complete packets as we can
				while (packet_buffer_size >= size_header) {
					// Parse the header to determine full packet size
					cceap_header_t *tmp_hdr = (cceap_header_t *)packet_buffer;
					int expected_size = size_header + (tmp_hdr->number_of_options * sizeof(options_t));
					
					// If we don't have a complete packet yet, break and wait for more data
					if (packet_buffer_size < expected_size) {
						if (verbose) {
							printf("Waiting for more data. Have %d bytes, need %d bytes\n",
								packet_buffer_size, expected_size);
						}
						break;
					}
					
					// We have a complete packet - process it
					hdr = (cceap_header_t *)packet_buffer;
					
					printf("received data (%d bytes):\n", expected_size);
					printf(" > time diff to prev pkt: ");
					print_time_diff(times_file);
					printf(" > sequence number:       %u\n", hdr->sequence_number);
					printf(" > destination length:    %d\n", hdr->destination_length);
					printf(" > dummy value:           %d\n", hdr->dummy);
					char dst_n_pad[DSTPADSIZ + 1] = { '\0' };
					strncpy(dst_n_pad, hdr->destination_and_padding, DSTPADSIZ);
					printf(" > destination + padding: %s\n", dst_n_pad);
					printf(" > number of options:     %d\n", hdr->number_of_options);
					
					if (hdr->number_of_options >= 1) {
						int i;
						
						options_array = (options_t *)(packet_buffer + sizeof(cceap_header_t));
						printf("    > options overview:\n");
						for (i = 0; i < hdr->number_of_options; i++) {
							printf("\t\t\t  #%d: (identifier: %hhu, type: %hhu, value: %hhu)\n",
								i + 1,
								(options_array + i)->opt_identifier,
								(options_array + i)->opt_type,
								(options_array + i)->opt_value);
						}
					}
					
					// Remove this packet from the buffer
					memmove(packet_buffer, packet_buffer + expected_size, 
							packet_buffer_size - expected_size);
					packet_buffer_size -= expected_size;
				}
			} else {
				printf("huh?!\n");
				exit(ERR_EXIT);
			}
		} else if (ret == 0) {
			/* do nothing */
		} else { /* ret = -1 */
			if (errno == EINTR)
				continue;
			else
				err(ERR_EXIT, "select");
		}
	}
	
	close(sockfd);
	return OK_EXIT;
}


