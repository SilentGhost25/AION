import { motion } from "framer-motion";
import { Link } from "wouter";
import { ArrowRight, BookOpen, Brain, CheckCircle2, FileText, ShieldCheck, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

export default function Home() {
  return (
    <div className="space-y-12 pb-10">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl bg-primary text-primary-foreground shadow-lg">
        <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1541339907198-e08756dedf3f?auto=format&fit=crop&q=80')] bg-cover bg-center opacity-10 mix-blend-overlay"></div>
        <div className="absolute inset-0 bg-gradient-to-r from-primary to-primary/80" />
        
        <div className="relative z-10 p-8 md:p-16 flex flex-col md:flex-row items-center gap-10">
          <div className="flex-1 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-foreground/10 border border-primary-foreground/20 text-sm font-medium backdrop-blur-sm">
              <ShieldCheck className="h-4 w-4" />
              <span>DSATM Official Tool</span>
            </div>
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight leading-tight font-serif">
              Institutional-Grade <br/> Question Papers in Minutes
            </h1>
            <p className="text-lg md:text-xl text-primary-foreground/80 max-w-2xl leading-relaxed">
              Ensure academic rigor, perfect CO-PO mapping, and NAAC compliance with our AI-powered paper generator. Built exclusively for DSATM faculty.
            </p>
            <div className="flex flex-wrap items-center gap-4 pt-4">
              <Link href="/generate">
                <Button size="lg" variant="secondary" className="font-semibold px-8 hover-elevate">
                  Generate Paper <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link href="/questions">
                <Button size="lg" variant="outline" className="bg-transparent border-primary-foreground/30 text-primary-foreground hover:bg-primary-foreground/10 hover:text-primary-foreground">
                  View Question Bank
                </Button>
              </Link>
            </div>
          </div>
          
          <div className="hidden lg:block w-1/3">
            <div className="relative aspect-[3/4] w-full max-w-sm ml-auto rounded-xl border border-primary-foreground/20 bg-primary-foreground/5 shadow-2xl backdrop-blur-sm p-6 overflow-hidden">
              <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-transparent via-primary-foreground/50 to-transparent" />
              <div className="space-y-4">
                <div className="h-6 w-1/2 bg-primary-foreground/20 rounded animate-pulse" />
                <div className="h-4 w-1/3 bg-primary-foreground/10 rounded animate-pulse" />
                <div className="space-y-2 mt-8">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="flex gap-4">
                      <div className="h-4 w-4 rounded-full bg-primary-foreground/20 shrink-0 mt-1" />
                      <div className="space-y-2 flex-1">
                        <div className="h-4 w-full bg-primary-foreground/10 rounded" />
                        <div className="h-4 w-4/5 bg-primary-foreground/10 rounded" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Model Section */}
      <section className="space-y-8">
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <h2 className="text-3xl font-bold text-foreground font-serif">The 4-Step AI Implementation Model</h2>
          <p className="text-muted-foreground text-lg">A systematic approach to standardized evaluations, reducing faculty workload while increasing quality.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            {
              step: "01",
              title: "Central Repository",
              desc: "Create a Central Question Bank Repository for the department.",
              icon: BookOpen,
              color: "text-blue-500",
              bg: "bg-blue-500/10"
            },
            {
              step: "02",
              title: "Faculty Contribution",
              desc: "Each faculty uploads ~200 questions per subject.",
              icon: Brain,
              color: "text-emerald-500",
              bg: "bg-emerald-500/10"
            },
            {
              step: "03",
              title: "Smart Tagging",
              desc: "Tag questions with CO, Bloom's Level, and Difficulty parameters.",
              icon: Zap,
              color: "text-amber-500",
              bg: "bg-amber-500/10"
            },
            {
              step: "04",
              title: "AI Generation",
              desc: "AI dynamically generates balanced question papers instantly.",
              icon: FileText,
              color: "text-purple-500",
              bg: "bg-purple-500/10"
            }
          ].map((item, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
            >
              <Card className="relative h-full border-muted/50 hover:border-primary/50 transition-colors">
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg ${item.bg} flex items-center justify-center mb-4`}>
                    <item.icon className={`h-6 w-6 ${item.color}`} />
                  </div>
                  <CardTitle className="flex items-center gap-2">
                    <span className="text-sm font-bold text-muted-foreground">Step {item.step}</span>
                  </CardTitle>
                  <CardTitle className="text-xl">{item.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground leading-relaxed">{item.desc}</p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      <Separator />

      {/* Benefits Section */}
      <section className="space-y-8">
        <h2 className="text-3xl font-bold text-center text-foreground font-serif">Key Benefits</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {[
            {
              title: "Academic Quality",
              items: ["Standardized evaluation across sections", "Balanced difficulty distribution", "Eliminates question repetition bias"]
            },
            {
              title: "Accreditation Support",
              items: ["Perfect NBA CO-PO mapping reports", "NAAC Criteria 2 compliance built-in", "Automated RBT (Bloom's) distribution tracking"]
            },
            {
              title: "Administrative Efficiency",
              items: ["Reduces paper setting time by 90%", "Streamlined approval workflow", "One-click formatting to VTU standards"]
            }
          ].map((benefit, idx) => (
            <Card key={idx} className="bg-card shadow-sm border-muted">
              <CardHeader>
                <CardTitle className="text-lg">{benefit.title}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {benefit.items.map((item, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                    <span className="text-muted-foreground text-sm">{item}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
