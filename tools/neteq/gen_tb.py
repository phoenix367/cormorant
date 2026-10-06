#!/usr/bin/env python3
"""gen_tb.py TOP.v JOBS.svh OUT.sv — a lockstep RTL-vs-netlist testbench for a
kernel top (HLS-style AXI ports): every input is shared, the AXI slaves react to
the RTL instance, and every output of the netlist is compared with the RTL's
each cycle (payloads while their VALID is high).

JOBS.svh is the body of the stimulus (AXI-Lite writes `wr(addr, data)` and
`start_and_wait(label)`); an optional "// @tasks" section before "// @jobs"
holds module-level tasks the jobs use."""
import re, sys
top_v, jobs, out = sys.argv[1:4]
src = open(top_v).read()
mod = re.search(r'^module (\w+)', src, re.M).group(1)
params = dict((m.group(1), m.group(2)) for m in re.finditer(r'parameter\s+(?:integer\s+)?(\w+)\s*=\s*([^,\n)]+)', src))
def width(expr):
    if not expr: return 1
    e = expr
    for k, v in params.items(): e = re.sub(r'\b%s\b' % k, v.strip(), e)
    hi, lo = e.split(':')
    return eval(hi) - eval(lo) + 1
ports = []
for m in re.finditer(r'^\s*(input|output)\s+wire\s*(?:\[([^\]]+)\])?\s*(\w+)', src, re.M):
    ports.append((m.group(1), width(m.group(2)), m.group(3)))
outs = [p for p in ports if p[0] == 'output']
ins  = [p for p in ports if p[0] == 'input' and p[2] not in ('ap_clk', 'ap_rst_n')]
rd_ports = sorted(set(re.match(r'm_axi_(\w+)_ARVALID', n).group(1) for d, w, n in outs if n.endswith('_ARVALID')))
wr_ports = sorted(set(re.match(r'm_axi_(\w+)_AWVALID', n).group(1) for d, w, n in outs if n.endswith('_AWVALID')))
L = []
L.append('`timescale 1ns/1ps\nmodule tbk;\n  logic clk = 0, rst_n = 0;\n  always #2 clk = ~clk;\n  int errs = 0;\n  int unsigned cyc = 0;')
for d, w, n in ins:  L.append('  logic [%d:0] i_%s = 0;' % (w - 1, n))
for d, w, n in outs: L.append('  logic [%d:0] r_%s, n_%s;' % (w - 1, n, n))
for inst, pre in ((mod, 'r'), ('N_' + mod, 'n')):
    conns = ['.ap_clk (clk)', '.ap_rst_n (rst_n)']
    conns += ['.%s (i_%s)' % (n, n) for d, w, n in ins]
    conns += ['.%s (%s_%s)' % (n, pre, n) for d, w, n in outs]
    L.append('  %s u_%s (\n    %s);' % (inst, pre, ',\n    '.join(conns)))
# compare
L.append('''  task automatic chk(input string what, input logic [255:0] a, input logic [255:0] b);
    if (a !== b) begin
      errs++;
      if (errs <= 30) $display("MISMATCH cyc %0d %s: rtl %h net %h", cyc, what, a, b);
    end
  endtask''')
L.append('  always @(posedge clk) if (rst_n) begin\n    cyc++;')
names = set(n for d, w, n in outs)
for d, w, n in outs:
    mm = re.match(r'(m_axi_\w+?)_(AW|AR|W)(\w+)$', n)
    sm = re.match(r's_axi_ctrl_(RDATA|RRESP|BRESP)$', n)
    if mm and mm.group(3) != 'VALID' and (mm.group(1) + '_' + mm.group(2) + 'VALID') in names:
        v = 'r_%s_%sVALID' % (mm.group(1), mm.group(2))
        if mm.group(2) == 'W' and mm.group(3) == 'DATA':
            L.append('    if (%s) begin logic [%d:0] m; for (int i = 0; i < %d; i++) m[8*i +: 8] = {8{r_%s_WSTRB[i]}};'
                     ' chk("%s", r_%s & m, n_%s & m); end' % (v, w - 1, w // 8, mm.group(1), n, n, n))
        else:
            L.append('    if (%s) chk("%s", r_%s, n_%s);' % (v, n, n, n))
    elif sm:
        v = 'r_s_axi_ctrl_RVALID' if sm.group(1).startswith('R') else 'r_s_axi_ctrl_BVALID'
        L.append('    if (%s) chk("%s", r_%s, n_%s);' % (v, n, n, n))
    else:
        L.append('    chk("%s", r_%s, n_%s);' % (n, n, n))
L.append('  end')
# slaves
for p in rd_ports:
    L.append('''  // read slave %(p)s
  typedef struct { longint unsigned addr; int beats; int unsigned due; } rq_%(p)s_t;
  rq_%(p)s_t rq_%(p)s[$];
  int rb_%(p)s = 0;
  always @(posedge clk) begin
    if (rst_n) begin
      if (r_m_axi_%(p)s_ARVALID && i_m_axi_%(p)s_ARREADY)
        rq_%(p)s.push_back('{r_m_axi_%(p)s_ARADDR, int'(r_m_axi_%(p)s_ARLEN) + 1, cyc + lat_min + ($urandom %% (lat_max - lat_min + 1))});
      if (i_m_axi_%(p)s_RVALID && r_m_axi_%(p)s_RREADY) begin
        rd_beats++;
        if (++rb_%(p)s == rq_%(p)s[0].beats) begin rb_%(p)s = 0; void'(rq_%(p)s.pop_front()); end
      end
    end
    #0.5;
    i_m_axi_%(p)s_ARREADY <= ($urandom %% 100) < p_ar;
    i_m_axi_%(p)s_RVALID  <= (rq_%(p)s.size() > 0) && (rq_%(p)s[0].due <= cyc) && (($urandom %% 100) < p_r);
    if (rq_%(p)s.size() > 0) begin
      i_m_axi_%(p)s_RDATA <= mem(rq_%(p)s[0].addr + 16 * rb_%(p)s);
      i_m_axi_%(p)s_RLAST <= (rb_%(p)s == rq_%(p)s[0].beats - 1);
    end
  end''' % {'p': p})
for p in wr_ports:
    L.append('''  // write slave %(p)s
  int wq_%(p)s[$];
  int unsigned bq_%(p)s[$];
  int wb_%(p)s = 0;
  always @(posedge clk) begin
    if (rst_n) begin
      if (r_m_axi_%(p)s_AWVALID && i_m_axi_%(p)s_AWREADY) wq_%(p)s.push_back(int'(r_m_axi_%(p)s_AWLEN) + 1);
      if (r_m_axi_%(p)s_WVALID && i_m_axi_%(p)s_WREADY) begin
        wr_beats++;
        if (++wb_%(p)s == wq_%(p)s[0]) begin wb_%(p)s = 0; void'(wq_%(p)s.pop_front()); bq_%(p)s.push_back(cyc + 10 + $urandom %% 50); end
      end
      if (i_m_axi_%(p)s_BVALID && r_m_axi_%(p)s_BREADY) void'(bq_%(p)s.pop_front());
    end
    #0.5;
    i_m_axi_%(p)s_AWREADY <= ($urandom %% 100) < p_aw;
    i_m_axi_%(p)s_WREADY  <= (wq_%(p)s.size() > 0) && (($urandom %% 100) < p_w);
    i_m_axi_%(p)s_BVALID  <= (bq_%(p)s.size() > 0) && (bq_%(p)s[0] <= cyc);
  end''' % {'p': p})
hdr = '''  int lat_min = 20, lat_max = 200, p_ar = 70, p_r = 80, p_aw = 70, p_w = 80;
  longint unsigned rd_beats = 0, wr_beats = 0;
  function automatic logic [127:0] mem(longint unsigned a);
    logic [127:0] v;
    for (int i = 0; i < 4; i++) v[32*i +: 32] = 32'(a * 2654435761 + i * 40503 + (a >> 7));
    return v;
  endfunction
  task automatic wr(input logic [7:0] a, input logic [31:0] d);
    @(negedge clk); i_s_axi_ctrl_AWVALID = 1; i_s_axi_ctrl_AWADDR = a;
    do @(posedge clk); while (!r_s_axi_ctrl_AWREADY);
    @(negedge clk); i_s_axi_ctrl_AWVALID = 0; i_s_axi_ctrl_WVALID = 1; i_s_axi_ctrl_WDATA = d; i_s_axi_ctrl_WSTRB = '1;
    do @(posedge clk); while (!r_s_axi_ctrl_WREADY);
    @(negedge clk); i_s_axi_ctrl_WVALID = 0; i_s_axi_ctrl_BREADY = 1;
    do @(posedge clk); while (!r_s_axi_ctrl_BVALID);
    @(negedge clk); i_s_axi_ctrl_BREADY = 0;
  endtask
  task automatic rd(input logic [7:0] a, output logic [31:0] d);
    @(negedge clk); i_s_axi_ctrl_ARVALID = 1; i_s_axi_ctrl_ARADDR = a;
    do @(posedge clk); while (!r_s_axi_ctrl_ARREADY);
    @(negedge clk); i_s_axi_ctrl_ARVALID = 0; i_s_axi_ctrl_RREADY = 1;
    do @(posedge clk); while (!r_s_axi_ctrl_RVALID);
    d = r_s_axi_ctrl_RDATA;
    @(negedge clk); i_s_axi_ctrl_RREADY = 0;
  endtask
  task automatic start_and_wait(input string label);
    logic [31:0] d;
    int unsigned t0;
    wr(8'h00, 1);
    t0 = cyc;
    do begin repeat (20) @(posedge clk); rd(8'h00, d); end while (!d[1] && cyc - t0 < 4000000);
    $display("job %s: %0d cycles, %0d errors so far", label, cyc - t0, errs);
  endtask'''
L.insert(1, hdr)
L.append('  initial begin\n    repeat (60) @(posedge clk);   // past the netlist GSR pulse\n    rst_n = 1;\n    repeat (8) @(posedge clk);')
jt = open(jobs).read()
tasks = ''
if '// @tasks' in jt:
    tasks, jt = jt.split('// @jobs', 1)
    tasks = tasks.replace('// @tasks', '')
L.insert(2, tasks)
L.append(jt)
L.append('    $display("DONE errors=%0d rd_beats=%0d wr_beats=%0d cycles=%0d", errs, rd_beats, wr_beats, cyc);\n    $finish;\n  end\nendmodule')
open(out, 'w').write('\n'.join(L) + '\n')
print(mod, 'read ports', rd_ports, 'write ports', wr_ports, len(ins), 'inputs', len(outs), 'outputs')
