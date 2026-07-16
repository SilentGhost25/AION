import { createContext, useContext, useState, ReactNode } from "react";

export interface TeacherSubject {
  id: string;
  code: string;
  name: string;
  department: string;
  semester: string;
  addedOn: string;
}

export interface TeacherProfile {
  name: string;
  designation: string;
  employeeId: string;
  department: string;
  subjects: TeacherSubject[];
}

interface ProfileContextType {
  profile: TeacherProfile;
  addSubject: (subject: Omit<TeacherSubject, "id" | "addedOn">) => void;
  removeSubject: (id: string) => void;
  updateProfile: (data: Partial<Omit<TeacherProfile, "subjects">>) => void;
}

const ProfileContext = createContext<ProfileContextType | null>(null);

const DEFAULT_PROFILE: TeacherProfile = {
  name: "Dr. John Doe",
  designation: "Associate Professor",
  employeeId: "DSATM-2019-047",
  department: "Artificial Intelligence & Machine Learning",
  subjects: [
    {
      id: "sub-1",
      code: "21AI51",
      name: "Machine Learning",
      department: "Artificial Intelligence & Machine Learning",
      semester: "5",
      addedOn: "2024-06-01",
    },
    {
      id: "sub-2",
      code: "21AI52",
      name: "Deep Learning",
      department: "Artificial Intelligence & Machine Learning",
      semester: "6",
      addedOn: "2024-06-01",
    },
    {
      id: "sub-3",
      code: "21AI55",
      name: "Data Structures and Algorithms",
      department: "Artificial Intelligence & Machine Learning",
      semester: "4",
      addedOn: "2024-07-10",
    },
  ],
};

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [profile, setProfile] = useState<TeacherProfile>(DEFAULT_PROFILE);

  const addSubject = (subject: Omit<TeacherSubject, "id" | "addedOn">) => {
    setProfile(prev => ({
      ...prev,
      subjects: [
        ...prev.subjects,
        {
          ...subject,
          id: `sub-${Date.now()}`,
          addedOn: new Date().toISOString().slice(0, 10),
        },
      ],
    }));
  };

  const removeSubject = (id: string) => {
    setProfile(prev => ({
      ...prev,
      subjects: prev.subjects.filter(s => s.id !== id),
    }));
  };

  const updateProfile = (data: Partial<Omit<TeacherProfile, "subjects">>) => {
    setProfile(prev => ({ ...prev, ...data }));
  };

  return (
    <ProfileContext.Provider value={{ profile, addSubject, removeSubject, updateProfile }}>
      {children}
    </ProfileContext.Provider>
  );
}

export function useProfile() {
  const ctx = useContext(ProfileContext);
  if (!ctx) throw new Error("useProfile must be used inside ProfileProvider");
  return ctx;
}
